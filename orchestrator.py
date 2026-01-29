#!/usr/bin/env python3
"""
orchestrator.py - Auto-Heal Pipeline Orchestrator

This is the main entry point that coordinates the entire auto-heal pipeline.
It delegates prompt building to specialized modules for better maintainability.

=== ARCHITECTURE ===

orchestrator.py (this file)
    └── Coordinates the pipeline
    └── Handles data loading, testing, and model selection

prompt_builders/
    ├── base.py            - Shared context extraction utilities
    ├── first_inference.py - Template A: Initial code generation
    └── improvement.py     - Template B: Iterative improvement

api_test.py
    └── LLM API client (Claude, OpenAI, Gemini)

=== PIPELINE FLOW ===

Phase 1: Initial Script Generation
    1. Load HTML samples + ground truth from CSV
    2. Split into 50% training / 50% test
    3. Build first inference prompt (Template A)
    4. Ask LLMs to generate extraction scripts
    5. Test and select best performing script

Phase 2: Iterative Improvement (up to 5 iterations)
    1. Build improvement prompt (Template B) with TRAINING failures only
    2. Ask LLM to fix the script
    3. Test and compare with previous versions
    4. Repeat until target accuracy or max iterations

=== KEY PRINCIPLES ===

1. Test Set Integrity - Test set is NEVER shown to LLM for feedback
2. Smart Context - Focused HTML snippets, not entire pages
3. Accuracy First - Model selection prioritizes correctness over speed
4. Separate Tracking - Order ID and subtotal accuracy tracked independently

Usage:
    python orchestrator.py

Output:
    - Generates candidate_script_*.py files during execution
    - Saves best script as fixed_script.py
"""

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import re

import api_test
from prompt_builders import first_inference, improvement
from prompt_builders.base import normalize_string
from script_utils import check_ground_truth_presence, display_ground_truth_presence, cleanup_and_save_final


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to CSV file containing HTML samples and ground truth
CSV_PATH = 'store_data/Store_OCP_Data.csv'

# Target pass rate (%) - stop improving if we hit this on test set
MIN_PASS_RATE = 70

# Maximum improvement iterations before giving up
MAX_ITERATIONS = 5

# =================================================================  ============
# MODELS TO TEST - Edit this list to control which models are tried
# =============================================================================
# Each entry is: (model_key, model_name, tier)
#   - model_key: Used for script naming and API routing
#   - model_name: Display name (from api_test)
#   - tier: "capable" (smarter) or "fast" (quicker)
#
# Available models:
#   ("claude",      api_test.CLAUDE_MODEL, "capable")  - Claude 3.5 Sonnet
#   ("openai",      api_test.OPENAI_MODEL, "capable")  - GPT-4.1
#   ("gemini",      api_test.GEMINI_MODEL, "capable")  - Gemini 2.5 Pro
#   ("claude_fast", api_test.CLAUDE_FAST,  "fast")     - Claude 3.5 Haiku
#   ("openai_fast", api_test.OPENAI_FAST,  "fast")     - GPT-4o Mini
#
# Just comment out or remove lines you don't want to test!
# =============================================================================

MODELS_TO_TEST = [
    ("claude_fast", api_test.CLAUDE_FAST, "fast"),     # Fast - try this first
    ("openai_fast", api_test.OPENAI_FAST, "fast"),     # Fast alternative
    # ("claude",      api_test.CLAUDE_MODEL, "capable"),  # Slower but smarter
    # ("openai",      api_test.OPENAI_MODEL, "capable"),  # Slower but smarter
    # ("gemini",      api_test.GEMINI_MODEL, "capable"),  # Google's model
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_data(csv_path=CSV_PATH):
    """
    Load HTML samples and ground truth from CSV file.
    
    Expected CSV columns:
        - STORE_ID: Unique identifier for the store/page
        - RAW_DOM: Full HTML content of the page
        - GROUND_TRUTH_ORDER_ID: Expected order number to extract
        - GROUND_TRUTH_SUBTOTAL: Expected subtotal amount to extract
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        List of dicts with keys: row_id, store_id, html_content, order_id, subtotal
    """
    csv.field_size_limit(sys.maxsize)
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append({
                'row_id': idx,
                'store_id': row['STORE_ID'],
                'html_content': row['RAW_DOM'],
                'order_id': row['GROUND_TRUTH_ORDER_ID'],
                'subtotal': row['GROUND_TRUTH_SUBTOTAL']
            })
    
    return rows


def split_train_test(rows, seed=42):
    """
    Split data into 50% training / 50% test with fixed seed.
    
    Args:
        rows: List of data rows to split
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_rows, test_rows)
    """
    shuffled = rows.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


# =============================================================================
# SCRIPT TESTING
# =============================================================================

def test_script(script_path, rows):
    """
    Test a generated script on a set of HTML samples.
    
    For each row:
    1. Write HTML to a temp file
    2. Run the script with the temp file as argument
    3. Parse JSON output and compare to ground truth
    4. Track timing for latency metrics
    
    Args:
        script_path: Path to the Python script to test
        rows: List of data rows with HTML and ground truth
    
    Returns:
        Tuple of (results_list, average_time_ms)
    """
    results = []
    total_time = 0
    temp_files = []
    
    try:
        for row in rows:
            row_id = row['row_id']
            html_content = row['html_content']
            expected_order_id = row['order_id']
            expected_subtotal = row['subtotal']
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
            temp_file.write(html_content)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            try:
                start = time.time()
                proc = subprocess.run(
                    [sys.executable, script_path, temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                elapsed = (time.time() - start) * 1000
                total_time += elapsed
                
                if proc.returncode != 0:
                    results.append({
                        'row_id': row_id,
                        'success': False, 
                        'order_id_correct': False,
                        'subtotal_correct': False,
                        'error': f"Exit code {proc.returncode}: {proc.stderr[:200]}",
                        'time': elapsed,
                        'expected_order_id': expected_order_id,
                        'expected_subtotal': expected_subtotal,
                        'extracted_order_id': None,
                        'extracted_subtotal': None
                    })
                    continue
                
                stdout = proc.stdout.strip()
                if not stdout:
                        results.append({
                        'row_id': row_id,
                            'success': False, 
                        'order_id_correct': False,
                        'subtotal_correct': False,
                        'error': "Empty output",
                        'time': elapsed,
                        'expected_order_id': expected_order_id,
                        'expected_subtotal': expected_subtotal,
                            'extracted_order_id': None,
                        'extracted_subtotal': None
                        })
                        continue
                    
                try:
                    extracted = json.loads(stdout)
                except json.JSONDecodeError:
                    results.append({
                        'row_id': row_id,
                        'success': False, 
                        'order_id_correct': False,
                        'subtotal_correct': False,
                        'error': f"Invalid JSON: {stdout[:100]}",
                        'time': elapsed,
                        'expected_order_id': expected_order_id,
                        'expected_subtotal': expected_subtotal,
                        'extracted_order_id': None,
                        'extracted_subtotal': None
                    })
                    continue
                
                # Extract values from result
                ext_order_id = None
                ext_subtotal = None
                if extracted and isinstance(extracted, list) and len(extracted) > 0:
                    if isinstance(extracted[0], dict):
                        ext_order_id = extracted[0].get('order_id')
                        ext_subtotal = extracted[0].get('subtotal')
                
                # Compare normalized values SEPARATELY
                exp_oid_norm = normalize_string(expected_order_id)
                exp_sub_norm = normalize_string(expected_subtotal)
                ext_oid_norm = normalize_string(ext_order_id)
                ext_sub_norm = normalize_string(ext_subtotal)
                
                order_id_correct = (exp_oid_norm == ext_oid_norm) and ext_oid_norm != ''
                subtotal_correct = (exp_sub_norm == ext_sub_norm) and ext_sub_norm != ''
                success = order_id_correct and subtotal_correct
                
                results.append({
                    'row_id': row_id,
                    'success': success,
                    'order_id_correct': order_id_correct,
                    'subtotal_correct': subtotal_correct,
                    'error': None if success else "Mismatch",
                    'time': elapsed,
                    'expected_order_id': expected_order_id,
                    'expected_subtotal': expected_subtotal,
                    'extracted_order_id': ext_order_id,
                    'extracted_subtotal': ext_subtotal
                })
            
            except subprocess.TimeoutExpired:
                results.append({
                    'row_id': row_id,
                    'success': False, 
                    'order_id_correct': False,
                    'subtotal_correct': False,
                    'error': "Timeout", 
                    'time': 10000,
                    'expected_order_id': expected_order_id,
                    'expected_subtotal': expected_subtotal,
                    'extracted_order_id': None,
                    'extracted_subtotal': None
                })
                total_time += 10000
            except Exception as e:
                results.append({
                    'row_id': row_id,
                    'success': False, 
                    'order_id_correct': False,
                    'subtotal_correct': False,
                    'error': str(e)[:150],
                    'time': 0,
                    'expected_order_id': expected_order_id,
                    'expected_subtotal': expected_subtotal,
                    'extracted_order_id': None,
                    'extracted_subtotal': None
                })
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass
    
    avg_time = total_time / len(rows) if rows else 0
    return results, avg_time


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(results):
    """
    Calculate comprehensive metrics from test results.
    
    Args:
        results: List of test result dicts
    
    Returns:
        Dict with metrics: total, order_id_correct, subtotal_correct, both_correct,
                          total_fields_correct, rates, and latency stats
    """
    total = len(results)
    oid_correct = sum(1 for r in results if r['order_id_correct'])
    sub_correct = sum(1 for r in results if r['subtotal_correct'])
    both_correct = sum(1 for r in results if r['success'])
    total_fields_correct = oid_correct + sub_correct
    
    latencies = [r.get('time', 0) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    return {
        'total': total,
        'order_id_correct': oid_correct,
        'subtotal_correct': sub_correct,
        'both_correct': both_correct,
        'total_fields_correct': total_fields_correct,
        'order_id_rate': (oid_correct * 100) // total if total else 0,
        'subtotal_rate': (sub_correct * 100) // total if total else 0,
        'both_rate': (both_correct * 100) // total if total else 0,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
    }


# =============================================================================
# CODE EXTRACTION
# =============================================================================

def extract_code_from_response(response):
    """
    Extract Python code from an LLM response.
    
    Handles: ```python...```, ```...```, or raw code
    
    Args:
        response: Raw text response from LLM
    
    Returns:
        Extracted Python code string
    """
    if not response:
        return ""
    
    response = response.strip()
    
    code_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    
    if 'import ' in response or 'def ' in response:
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        return response.strip()
    
    return response.strip()


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_results(results, title="Results", max_rows=20):
    """Display detailed per-row results in a formatted table."""
    print(f"\n   {title}:")
    print(f"   {'Row':<10} | {'OID':<4} | {'SUB':<4} | {'ms':<6} | {'Expected Order ID':<18} | {'Got Order ID':<18} | {'Exp Sub':<10} | {'Got Sub':<10}")
    print(f"   {'-'*10} | {'-'*4} | {'-'*4} | {'-'*6} | {'-'*18} | {'-'*18} | {'-'*10} | {'-'*10}")
    
    for r in results[:max_rows]:
        oid_status = "✅" if r['order_id_correct'] else "❌"
        sub_status = "✅" if r['subtotal_correct'] else "❌"
        latency = f"{r.get('time', 0):.0f}"
        
        exp_oid = str(r['expected_order_id'])[:18] if r['expected_order_id'] else 'N/A'
        got_oid = str(r.get('extracted_order_id') or 'NONE')[:18]
        exp_sub = str(r['expected_subtotal'])[:10] if r['expected_subtotal'] else 'N/A'
        got_sub = str(r.get('extracted_subtotal') or 'NONE')[:10]
        
        print(f"   row_{r['row_id']:<5} | {oid_status:<4} | {sub_status:<4} | {latency:<6} | {exp_oid:<18} | {got_oid:<18} | {exp_sub:<10} | {got_sub:<10}")
    
    if len(results) > max_rows:
        print(f"   ... and {len(results) - max_rows} more rows")


def display_metrics(train_metrics, test_metrics, title="Summary"):
    """Display summary comparison table of train vs test metrics."""
    print(f"\n   {title}:")
    print(f"   ┌────────────────────┬──────────────────┬──────────────────┐")
    print(f"   │ Metric             │ Training         │ Test (Holdout)   │")
    print(f"   ├────────────────────┼──────────────────┼──────────────────┤")
    print(f"   │ Order ID Correct   │ {train_metrics['order_id_correct']:>3}/{train_metrics['total']:<3} ({train_metrics['order_id_rate']:>3}%)   │ {test_metrics['order_id_correct']:>3}/{test_metrics['total']:<3} ({test_metrics['order_id_rate']:>3}%)   │")
    print(f"   │ Subtotal Correct   │ {train_metrics['subtotal_correct']:>3}/{train_metrics['total']:<3} ({train_metrics['subtotal_rate']:>3}%)   │ {test_metrics['subtotal_correct']:>3}/{test_metrics['total']:<3} ({test_metrics['subtotal_rate']:>3}%)   │")
    print(f"   │ Both Correct       │ {train_metrics['both_correct']:>3}/{train_metrics['total']:<3} ({train_metrics['both_rate']:>3}%)   │ {test_metrics['both_correct']:>3}/{test_metrics['total']:<3} ({test_metrics['both_rate']:>3}%)   │")
    print(f"   │ Total Fields       │ {train_metrics['total_fields_correct']:>3}/{train_metrics['total']*2:<3}          │ {test_metrics['total_fields_correct']:>3}/{test_metrics['total']*2:<3}          │")
    print(f"   ├────────────────────┼──────────────────┼──────────────────┤")
    print(f"   │ Avg Latency (ms)   │ {train_metrics['avg_latency']:>10.1f}       │ {test_metrics['avg_latency']:>10.1f}       │")
    print(f"   │ Min Latency (ms)   │ {train_metrics['min_latency']:>10.1f}       │ {test_metrics['min_latency']:>10.1f}       │")
    print(f"   │ Max Latency (ms)   │ {train_metrics['max_latency']:>10.1f}       │ {test_metrics['max_latency']:>10.1f}       │")
    print(f"   └────────────────────┴──────────────────┴──────────────────┘")


# =============================================================================
# MODEL SELECTION
# =============================================================================

def select_best_model(model_results):
    """
    Select best model: accuracy first (total correct fields on TEST), then latency.
    
    Args:
        model_results: List of dicts with model performance data
    
    Returns:
        The best model result dict, or None if no results
    """
    if not model_results:
        return None
    
    sorted_results = sorted(
        model_results,
        key=lambda x: (-x['test_metrics']['total_fields_correct'], x['avg_time'])
    )
    
    return sorted_results[0]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for the Auto-Heal Pipeline."""
    
    print("\n" + "=" * 120)
    print("🚀 AUTO-HEAL PIPELINE v3.0 - Modular Architecture")
    print("   Order ID + Subtotal tracked separately | Best model = max total correct fields on TEST")
    print("=" * 120)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print(f"\n📂 Loading data from CSV: {CSV_PATH}")
    rows = load_csv_data()
    print(f"   Found {len(rows)} rows in CSV")
    
    print("\n🎲 Splitting train/test (50/50, fixed seed)...")
    train_rows, test_rows = split_train_test(rows, seed=42)
    print(f"   Training: {len(train_rows)} rows (used for feedback)")
    print(f"   Test:     {len(test_rows)} rows (holdout - NEVER shown to model)")
    
    # =========================================================================
    # GROUND TRUTH PRESENCE CHECK
    # =========================================================================
    
    print("\n" + "-" * 120)
    print("🔍 Checking Ground Truth Presence in HTML")
    print("-" * 120)
    
    train_presence = check_ground_truth_presence(train_rows, "Training")
    test_presence = check_ground_truth_presence(test_rows, "Test")
    display_ground_truth_presence(train_presence, test_presence)
    
    # =========================================================================
    # BASELINE TEST
    # =========================================================================
    
    print("\n" + "-" * 120)
    print("📊 BASELINE: Testing bad_script.py")
    print("-" * 120)
    
    if os.path.exists('bad_script.py'):
        baseline_train_results, _ = test_script('bad_script.py', train_rows[:20])
        baseline_test_results, _ = test_script('bad_script.py', test_rows[:20])
        
        baseline_train_metrics = calculate_metrics(baseline_train_results)
        baseline_test_metrics = calculate_metrics(baseline_test_results)
        
        display_results(baseline_train_results, "Baseline - Training Set", max_rows=10)
        display_results(baseline_test_results, "Baseline - Test Set", max_rows=10)
        display_metrics(baseline_train_metrics, baseline_test_metrics, "Baseline Summary")
    else:
        print("   ⚠️  bad_script.py not found, skipping baseline")
    
    # =========================================================================
    # DETERMINE MODELS TO TEST
    # =========================================================================
    
    # Use the MODELS_TO_TEST list from configuration section
    models_to_test = MODELS_TO_TEST
    
    if not models_to_test:
        print("\n❌ No models configured in MODELS_TO_TEST. Please add at least one model.")
        return
    
    print(f"\n📋 Models to test: {', '.join(m[0].upper() for m in models_to_test)}")
    
    # =========================================================================
    # PHASE 1: INITIAL SCRIPT GENERATION
    # =========================================================================
    
    print("\n" + "-" * 120)
    print("📝 Building first inference prompt (Template A) - TRAINING DATA ONLY")
    print("-" * 120)
    
    # Use the modular prompt builder
    prompt = first_inference.build_prompt(train_rows)
    print(f"   Prompt size: {len(prompt)} chars")
    print(f"   Using {min(first_inference.MAX_TRAINING_EXAMPLES, len(train_rows))} examples with context windows")
            
    print("\n" + "-" * 120)
    print(f"🤖 PHASE 1: Initial Script Generation from {len(models_to_test)} Models (🧠 Capable + ⚡ Fast)")
    print("-" * 120)
    
    model_results = []
    test_sample = test_rows[:50] if len(test_rows) > 50 else test_rows
    train_sample = train_rows[:50] if len(train_rows) > 50 else train_rows
    
    for model_type, model_name, model_tier in models_to_test:
        try:
            tier_icon = "⚡" if model_tier == "fast" else "🧠"
            print(f"\n   Calling {model_type.upper()} ({model_name}) {tier_icon}...")
            
            # Call the appropriate API
            if model_tier == "fast":
                base_type = model_type.replace("_fast", "")
                if base_type == "claude":
                    response = api_test.call_claude_fast(prompt, max_tokens=4000)
                elif base_type == "openai":
                    response = api_test.call_openai_fast(prompt, max_tokens=4000)
                elif base_type == "gemini":
                    response = api_test.call_gemini_fast(prompt, max_tokens=4000)
                else:
                    response = api_test.call_llm(prompt, max_tokens=4000, model_type=base_type)
            else:
                response = api_test.call_llm(prompt, max_tokens=4000, model_type=model_type)
            
            if not response or len(response.strip()) == 0:
                print(f"   ⚠️  {model_type.upper()} returned empty response")
                continue
            
            code = extract_code_from_response(response)
            if not code or len(code.strip()) == 0:
                print(f"   ⚠️  {model_type.upper()} code extraction failed")
                continue
            
            script_path = f'candidate_script_{model_type}.py'
            with open(script_path, 'w') as f:
                f.write(code)
            print(f"   ✅ {model_type.upper()} script saved ({len(code)} chars)")
            
            # Test on both sets
            print(f"   Testing {model_type.upper()} script...")
            train_results, train_time = test_script(script_path, train_sample)
            test_results, test_time = test_script(script_path, test_sample)
            avg_time = (train_time + test_time) / 2
            
            train_metrics = calculate_metrics(train_results)
            test_metrics = calculate_metrics(test_results)
            
            model_results.append({
                'model_type': model_type,
                'model_name': model_name,
                'model_tier': model_tier,
                'script_path': script_path,
                'train_results': train_results,
                'test_results': test_results,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'avg_time': avg_time
            })
            
            tier_label = f"{tier_icon} {model_tier.upper()}"
            display_results(train_results, f"{model_type.upper()} ({tier_label}) - Training Set", max_rows=10)
            display_results(test_results, f"{model_type.upper()} ({tier_label}) - Test Set (Holdout)", max_rows=10)
            display_metrics(train_metrics, test_metrics, f"{model_type.upper()} ({tier_label}) Summary")
            
        except Exception as e:
            print(f"   ❌ {model_type.upper()} failed: {e}")
    
    if not model_results:
        print("\n❌ All models failed. Exiting.")
        return
        
    # =========================================================================
    # SELECT BEST MODEL
    # =========================================================================
    
    best = select_best_model(model_results)
    best_tier_icon = "⚡" if best.get('model_tier') == "fast" else "🧠"
    
    print("\n" + "-" * 120)
    print(f"🏆 BEST MODEL (by total correct fields on TEST): {best['model_type'].upper()} ({best['model_name']}) {best_tier_icon}")
    print(f"   Model Tier:      {best.get('model_tier', 'capable').upper()}")
    print(f"   Test Order IDs:  {best['test_metrics']['order_id_correct']}/{best['test_metrics']['total']} ({best['test_metrics']['order_id_rate']}%)")
    print(f"   Test Subtotals:  {best['test_metrics']['subtotal_correct']}/{best['test_metrics']['total']} ({best['test_metrics']['subtotal_rate']}%)")
    print(f"   Test Both:       {best['test_metrics']['both_correct']}/{best['test_metrics']['total']} ({best['test_metrics']['both_rate']}%)")
    print(f"   Total Fields:    {best['test_metrics']['total_fields_correct']}/{best['test_metrics']['total']*2}")
    print(f"   Avg Latency:     {best['avg_time']:.1f}ms")
    print("-" * 120)
    
    # Check if we've hit target
    if best['test_metrics']['both_rate'] >= MIN_PASS_RATE:
        print(f"\n✅ Already achieved {best['test_metrics']['both_rate']}% on test set (target: {MIN_PASS_RATE}%)!")
        final_script = cleanup_and_save_final(
            best['script_path'],
            best['model_type'],
            0  # No iterations needed
        )
        print("=" * 120 + "\n")
        return
    
    # =========================================================================
    # PHASE 2: ITERATIVE IMPROVEMENT
    # =========================================================================
    
    print("\n" + "-" * 120)
    print("🔄 PHASE 2: Iterative Improvement (feedback from TRAINING failures only)")
    print("-" * 120)
    
    current_script = best['script_path']
    current_train_results = best['train_results']
    current_test_results = best['test_results']
    current_test_metrics = best['test_metrics']
    current_avg_time = best['avg_time']
    
    best_iteration = {
        'script': current_script,
        'test_metrics': current_test_metrics,
        'avg_time': current_avg_time,
        'iteration': 0
    }
    
    all_iterations = [{
        'iteration': 0,
        'script': current_script,
        'test_metrics': current_test_metrics,
        'avg_time': current_avg_time
    }]
    
    iteration = 1
    while current_test_metrics['both_rate'] < MIN_PASS_RATE and iteration <= MAX_ITERATIONS:
        print(f"\n🔄 Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"   Current Test: OID={current_test_metrics['order_id_rate']}%, SUB={current_test_metrics['subtotal_rate']}%, Both={current_test_metrics['both_rate']}%")
        print("   Building improvement prompt (Template B)...")
        
        # Use the modular prompt builder
        feedback_prompt = improvement.build_prompt(
            current_script, current_train_results, current_test_results, train_sample, iteration
        )
        print(f"   Prompt size: {len(feedback_prompt)} chars")
        
        try:
            model_type = best['model_type']
            model_tier = best.get('model_tier', 'capable')
            print(f"   Calling {model_type.upper()} for improvement...")
            
            if model_tier == "fast":
                base_type = model_type.replace("_fast", "")
                if base_type == "claude":
                    response = api_test.call_claude_fast(feedback_prompt, max_tokens=4000)
                elif base_type == "openai":
                    response = api_test.call_openai_fast(feedback_prompt, max_tokens=4000)
                elif base_type == "gemini":
                    response = api_test.call_gemini_fast(feedback_prompt, max_tokens=4000)
                else:
                    response = api_test.call_llm(feedback_prompt, max_tokens=4000, model_type=base_type)
            else:
                response = api_test.call_llm(feedback_prompt, max_tokens=4000, model_type=model_type)
            
            if not response or len(response.strip()) == 0:
                print("   ⚠️  Empty response, stopping iteration")
                break
            
            code = extract_code_from_response(response)
            if not code or len(code.strip()) == 0:
                print("   ⚠️  Code extraction failed, stopping iteration")
                break
            
            improved_script = f"candidate_script_{best['model_type']}_v{iteration + 1}.py"
            with open(improved_script, 'w') as f:
                f.write(code)
            print(f"   ✅ Improved script: {improved_script}")
            
            # Test improved script
            new_train_results, new_train_time = test_script(improved_script, train_sample)
            new_test_results, new_test_time = test_script(improved_script, test_sample)
            new_avg_time = (new_train_time + new_test_time) / 2
            
            new_train_metrics = calculate_metrics(new_train_results)
            new_test_metrics = calculate_metrics(new_test_results)
            
            display_results(new_train_results, f"Iteration {iteration + 1} - Training Set", max_rows=10)
            display_results(new_test_results, f"Iteration {iteration + 1} - Test Set", max_rows=10)
            display_metrics(new_train_metrics, new_test_metrics, f"Iteration {iteration + 1} Summary")
            
            all_iterations.append({
                'iteration': iteration,
                'script': improved_script,
                'test_metrics': new_test_metrics,
                'avg_time': new_avg_time
            })
            
            # Track best (accuracy first, latency second)
            if new_test_metrics['total_fields_correct'] > best_iteration['test_metrics']['total_fields_correct'] or \
               (new_test_metrics['total_fields_correct'] == best_iteration['test_metrics']['total_fields_correct'] and new_avg_time < best_iteration['avg_time']):
                best_iteration = {
                    'script': improved_script,
                    'test_metrics': new_test_metrics,
                    'avg_time': new_avg_time,
                    'iteration': iteration
                }
                print(f"   🌟 New best! Total fields: {new_test_metrics['total_fields_correct']}/{new_test_metrics['total']*2}, Latency: {new_avg_time:.1f}ms")
            
            current_script = improved_script
            current_train_results = new_train_results
            current_test_results = new_test_results
            current_test_metrics = new_test_metrics
            current_avg_time = new_avg_time
            
            iteration += 1
            
        except Exception as e:
            print(f"   ❌ Improvement failed: {e}")
            break
        
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 120)
    print("📊 FINAL RESULTS")
    print("=" * 120)
    
    print("\n   📈 All Iterations Comparison (sorted by accuracy first, then latency):")
    print(f"   {'Iter':<6} | {'OID':<10} | {'SUB':<10} | {'Total Fields':<14} | {'Avg Latency':<12} | {'Status':<10}")
    print(f"   {'-'*6} | {'-'*10} | {'-'*10} | {'-'*14} | {'-'*12} | {'-'*10}")
    
    for it in all_iterations:
        is_best = it['iteration'] == best_iteration['iteration']
        status = "⭐ BEST" if is_best else ""
        m = it['test_metrics']
        print(f"   {it['iteration']:<6} | {m['order_id_correct']:>3}/{m['total']:<3} ({m['order_id_rate']:>2}%) | {m['subtotal_correct']:>3}/{m['total']:<3} ({m['subtotal_rate']:>2}%) | {m['total_fields_correct']:>5}/{m['total']*2:<5}      | {it['avg_time']:>8.1f}ms   | {status:<10}")
    
    print(f"\n   🏆 Winner: Iteration {best_iteration['iteration']} (0 = initial)")
    print(f"   Selection: Max total fields = {best_iteration['test_metrics']['total_fields_correct']}, then lowest latency = {best_iteration['avg_time']:.1f}ms")
    print(f"\n   Best Script: {best_iteration['script']}")
    print(f"   Test Order IDs:  {best_iteration['test_metrics']['order_id_correct']}/{best_iteration['test_metrics']['total']} ({best_iteration['test_metrics']['order_id_rate']}%)")
    print(f"   Test Subtotals:  {best_iteration['test_metrics']['subtotal_correct']}/{best_iteration['test_metrics']['total']} ({best_iteration['test_metrics']['subtotal_rate']}%)")
    print(f"   Test Both:       {best_iteration['test_metrics']['both_correct']}/{best_iteration['test_metrics']['total']} ({best_iteration['test_metrics']['both_rate']}%)")
    print(f"   Total Fields:    {best_iteration['test_metrics']['total_fields_correct']}/{best_iteration['test_metrics']['total']*2}")
    print(f"   Avg Latency:     {best_iteration['avg_time']:.1f}ms")
    
    if best_iteration['test_metrics']['both_rate'] >= MIN_PASS_RATE:
        print(f"\n✅ SUCCESS! Achieved {best_iteration['test_metrics']['both_rate']}% on test set (target: {MIN_PASS_RATE}%)")
    else:
        print(f"\n⚠️  Stopped at {best_iteration['test_metrics']['both_rate']}% on test set (target: {MIN_PASS_RATE}%)")
    
    # Clean up candidate scripts and save final
    final_script = cleanup_and_save_final(
        best_iteration['script'],
        best['model_type'],
        best_iteration['iteration']
    )
    
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()

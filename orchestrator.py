#!/usr/bin/env python3
"""
orchestrator.py - Auto-Heal Pipeline Orchestrator (Multi-Store Support)

This is the main entry point that coordinates the entire auto-heal pipeline.
It delegates prompt building to specialized modules for better maintainability.

=== ARCHITECTURE ===

orchestrator.py (this file)
    └── Coordinates the pipeline
    └── Handles data loading, testing, and model selection
    └── Processes EACH store independently

prompt_builders/
    ├── base.py            - Shared context extraction utilities
    ├── first_inference.py - Template A: Initial code generation
    └── improvement.py     - Template B: Iterative improvement

api_test.py
    └── LLM API client (Claude, OpenAI, Gemini)

=== PIPELINE FLOW (PER STORE) ===

For EACH unique STORE_ID in the CSV:

Phase 1: Initial Script Generation
    1. Filter rows for this store
    2. Split into 50% training / 50% test
    3. Build first inference prompt (Template A)
    4. Ask LLMs to generate extraction scripts
    5. Test and select best performing script

Phase 2: Iterative Improvement (configurable iterations)
    1. Build improvement prompt (Template B) with TRAINING failures only
    2. Ask LLM to fix the script
    3. Test and compare with previous versions
    4. Repeat until target accuracy or max iterations

=== KEY PRINCIPLES ===

1. Per-Store Processing - Each store gets its own optimized script
2. Test Set Integrity - Test set is NEVER shown to LLM for feedback
3. Smart Context - Focused HTML snippets, not entire pages
4. Accuracy First - Model selection prioritizes correctness over speed
5. Separate Tracking - Order ID and subtotal accuracy tracked independently

Usage:
    python orchestrator.py

Output:
    - Generates final_{store_id}_{model}_{iter}.py for each store
    - Displays aggregated results across all stores
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
# CSV should have columns: STORE_ID, RAW_DOM, GROUND_TRUTH_ORDER_ID, GROUND_TRUTH_SUBTOTAL
CSV_PATH = 'store_data/Store_OCP_Data.csv'

# Target pass rate (%) - stop improving if we hit this on test set
MIN_PASS_RATE = 70

# Maximum improvement iterations per store
# Set to 1 for quick testing across many stores (lower API costs)
# Set to 5 for thorough optimization on fewer stores
MAX_ITERATIONS = 1

# Minimum samples required per store to process
# Stores with fewer samples will be skipped (can't do 50/50 split)
MIN_SAMPLES_PER_STORE = 2

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


def get_unique_stores(rows):
    """
    Get a list of unique store IDs from the data.
    
    Args:
        rows: List of data rows with 'store_id' key
    
    Returns:
        Sorted list of unique store IDs
    """
    store_ids = set(row['store_id'] for row in rows)
    return sorted(store_ids)


def filter_by_store(rows, store_id):
    """
    Filter rows to only include those from a specific store.
    
    Args:
        rows: List of all data rows
        store_id: The store ID to filter for
    
    Returns:
        List of rows matching the store_id
    """
    return [r for r in rows if r['store_id'] == store_id]


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
# SINGLE STORE PROCESSING
# =============================================================================

def process_single_store(store_id, store_rows, models_to_test, verbose=True):
    """
    Process a single store through the full pipeline.
    
    Args:
        store_id: The store identifier
        store_rows: List of data rows for this store
        models_to_test: List of (model_key, model_name, tier) tuples
        verbose: If True, print detailed output
    
    Returns:
        Dict with store results or None if processing failed
    """
    
    if verbose:
        print(f"\n   📂 Store {store_id}: {len(store_rows)} samples")
    
    # Check minimum samples
    if len(store_rows) < MIN_SAMPLES_PER_STORE:
        if verbose:
            print(f"   ⚠️  Skipping - only {len(store_rows)} sample(s), need at least {MIN_SAMPLES_PER_STORE}")
        return None
    
    # Split this store's data 50/50
    train_rows, test_rows = split_train_test(store_rows, seed=42)
    if verbose:
        print(f"   Training: {len(train_rows)} | Test: {len(test_rows)}")
    
    # Check ground truth presence
    train_presence = check_ground_truth_presence(train_rows, "Training")
    test_presence = check_ground_truth_presence(test_rows, "Test")
    
    if verbose:
        print(f"   Ground Truth: OID={train_presence['order_id_pct']}%/{test_presence['order_id_pct']}%, SUB={train_presence['subtotal_pct']}%/{test_presence['subtotal_pct']}%")
    
    # =========================================================================
    # PHASE 1: INITIAL SCRIPT GENERATION
    # =========================================================================
    
    prompt = first_inference.build_prompt(train_rows)
    
    model_results = []
    test_sample = test_rows[:50] if len(test_rows) > 50 else test_rows
    train_sample = train_rows[:50] if len(train_rows) > 50 else train_rows
    
    for model_type, model_name, model_tier in models_to_test:
        try:
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
                continue
            
            code = extract_code_from_response(response)
            if not code or len(code.strip()) == 0:
                continue
            
            script_path = f'candidate_script_{store_id}_{model_type}.py'
            with open(script_path, 'w') as f:
                f.write(code)
            
            # Test on both sets
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
            
        except Exception as e:
            if verbose:
                print(f"   ❌ {model_type.upper()} failed: {e}")
    
    if not model_results:
        if verbose:
            print(f"   ❌ All models failed for store {store_id}")
        return None
    
    # =========================================================================
    # SELECT BEST MODEL
    # =========================================================================
    
    best = select_best_model(model_results)
    
    # Check if we've hit target - skip iterations if so
    if best['test_metrics']['both_rate'] >= MIN_PASS_RATE:
        final_script = cleanup_and_save_final(
            best['script_path'],
            store_id,
            best['model_type'],
            0
        )
        return {
            'store_id': store_id,
            'samples': len(store_rows),
            'train_count': len(train_rows),
            'test_count': len(test_rows),
            'best_model': best['model_type'],
            'iterations': 0,
            'test_oid_rate': best['test_metrics']['order_id_rate'],
            'test_sub_rate': best['test_metrics']['subtotal_rate'],
            'test_both_rate': best['test_metrics']['both_rate'],
            'total_fields': best['test_metrics']['total_fields_correct'],
            'max_fields': best['test_metrics']['total'] * 2,
            'avg_latency': best['avg_time'],
            'final_script': final_script,
            'status': 'success'
        }
    
    # =========================================================================
    # PHASE 2: ITERATIVE IMPROVEMENT
    # =========================================================================
    
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
    
    iteration = 1
    while current_test_metrics['both_rate'] < MIN_PASS_RATE and iteration <= MAX_ITERATIONS:
        feedback_prompt = improvement.build_prompt(
            current_script, current_train_results, current_test_results, train_sample, iteration
        )
        
        try:
            model_type = best['model_type']
            model_tier = best.get('model_tier', 'capable')
            
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
                break
            
            code = extract_code_from_response(response)
            if not code or len(code.strip()) == 0:
                break
            
            improved_script = f"candidate_script_{store_id}_{best['model_type']}_v{iteration + 1}.py"
            with open(improved_script, 'w') as f:
                f.write(code)
            
            # Test improved script
            new_train_results, new_train_time = test_script(improved_script, train_sample)
            new_test_results, new_test_time = test_script(improved_script, test_sample)
            new_avg_time = (new_train_time + new_test_time) / 2
            
            new_test_metrics = calculate_metrics(new_test_results)
            
            # Track best (accuracy first, latency second)
            if new_test_metrics['total_fields_correct'] > best_iteration['test_metrics']['total_fields_correct'] or \
               (new_test_metrics['total_fields_correct'] == best_iteration['test_metrics']['total_fields_correct'] and new_avg_time < best_iteration['avg_time']):
                best_iteration = {
                    'script': improved_script,
                    'test_metrics': new_test_metrics,
                    'avg_time': new_avg_time,
                    'iteration': iteration
                }
            
            current_script = improved_script
            current_train_results = new_train_results
            current_test_results = new_test_results
            current_test_metrics = new_test_metrics
            current_avg_time = new_avg_time
            
            iteration += 1
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Iteration {iteration} failed: {e}")
            break
    
    # Clean up candidate scripts and save final
    final_script = cleanup_and_save_final(
        best_iteration['script'],
        store_id,
        best['model_type'],
        best_iteration['iteration']
    )
    
    if verbose:
        tier_icon = "⚡" if best.get('model_tier') == "fast" else "🧠"
        print(f"   ✅ {best['model_type'].upper()} {tier_icon} | OID: {best_iteration['test_metrics']['order_id_rate']}% | SUB: {best_iteration['test_metrics']['subtotal_rate']}% | Iter: {best_iteration['iteration']} | {best_iteration['avg_time']:.0f}ms")
    
    return {
        'store_id': store_id,
        'samples': len(store_rows),
        'train_count': len(train_rows),
        'test_count': len(test_rows),
        'best_model': best['model_type'],
        'iterations': best_iteration['iteration'],
        'test_oid_rate': best_iteration['test_metrics']['order_id_rate'],
        'test_sub_rate': best_iteration['test_metrics']['subtotal_rate'],
        'test_both_rate': best_iteration['test_metrics']['both_rate'],
        'total_fields': best_iteration['test_metrics']['total_fields_correct'],
        'max_fields': best_iteration['test_metrics']['total'] * 2,
        'avg_latency': best_iteration['avg_time'],
        'final_script': final_script,
        'status': 'success'
    }


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

def display_store_summary(all_store_results):
    """
    Display aggregated results across all stores.
    
    Args:
        all_store_results: List of store result dicts
    """
    # Filter successful results
    successful = [r for r in all_store_results if r is not None and r.get('status') == 'success']
    skipped = len(all_store_results) - len(successful)
    
    if not successful:
        print("\n   ❌ No stores were successfully processed.")
        return
    
    print(f"\n   📊 Store Results Summary ({len(successful)} stores processed, {skipped} skipped):")
    print(f"   {'Store ID':<15} | {'Samples':<8} | {'Model':<12} | {'OID %':<7} | {'SUB %':<7} | {'Both %':<7} | {'Fields':<10} | {'Latency':<10} | {'Iter':<5}")
    print(f"   {'-'*15} | {'-'*8} | {'-'*12} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*10} | {'-'*10} | {'-'*5}")
    
    for r in sorted(successful, key=lambda x: -x['total_fields']):
        store_display = str(r['store_id'])[:15]
        print(f"   {store_display:<15} | {r['samples']:<8} | {r['best_model']:<12} | {r['test_oid_rate']:>5}% | {r['test_sub_rate']:>5}% | {r['test_both_rate']:>5}% | {r['total_fields']:>3}/{r['max_fields']:<5} | {r['avg_latency']:>7.1f}ms | {r['iterations']:<5}")
    
    # Aggregate statistics
    total_fields = sum(r['total_fields'] for r in successful)
    max_fields = sum(r['max_fields'] for r in successful)
    avg_oid = sum(r['test_oid_rate'] for r in successful) / len(successful)
    avg_sub = sum(r['test_sub_rate'] for r in successful) / len(successful)
    avg_both = sum(r['test_both_rate'] for r in successful) / len(successful)
    avg_latency = sum(r['avg_latency'] for r in successful) / len(successful)
    
    print(f"   {'-'*15} | {'-'*8} | {'-'*12} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*10} | {'-'*10} | {'-'*5}")
    print(f"   {'AVERAGE':<15} | {'':<8} | {'':<12} | {avg_oid:>5.1f}% | {avg_sub:>5.1f}% | {avg_both:>5.1f}% | {total_fields:>3}/{max_fields:<5} | {avg_latency:>7.1f}ms |")
    
    # Model distribution
    model_counts = {}
    for r in successful:
        m = r['best_model']
        model_counts[m] = model_counts.get(m, 0) + 1
    
    print(f"\n   🏆 Best Model Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = (count * 100) // len(successful)
        print(f"      {model}: {count} stores ({pct}%)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for the Multi-Store Auto-Heal Pipeline."""
    
    print("\n" + "=" * 120)
    print("🚀 AUTO-HEAL PIPELINE v4.0 - Multi-Store Support")
    print(f"   Per-store processing | MAX_ITERATIONS={MAX_ITERATIONS} | MIN_SAMPLES={MIN_SAMPLES_PER_STORE}")
    print("=" * 120)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print(f"\n📂 Loading data from CSV: {CSV_PATH}")
    all_rows = load_csv_data()
    print(f"   Found {len(all_rows)} total rows")
    
    # Get unique stores
    store_ids = get_unique_stores(all_rows)
    print(f"   Found {len(store_ids)} unique store(s)")
    
    # =========================================================================
    # VALIDATE MODELS
    # =========================================================================
    
    models_to_test = MODELS_TO_TEST
    if not models_to_test:
        print("\n❌ No models configured in MODELS_TO_TEST. Please add at least one model.")
        return
    
    print(f"   Models to test: {', '.join(m[0].upper() for m in models_to_test)}")
    
    # =========================================================================
    # BASELINE TEST (optional - on first store)
    # =========================================================================
    
    if os.path.exists('bad_script.py') and len(store_ids) > 0:
        print("\n" + "-" * 120)
        print("📊 BASELINE: Testing bad_script.py on first store")
        print("-" * 120)
        
        first_store_rows = filter_by_store(all_rows, store_ids[0])
        train_rows, test_rows = split_train_test(first_store_rows, seed=42)
        
        baseline_train_results, _ = test_script('bad_script.py', train_rows[:20])
        baseline_test_results, _ = test_script('bad_script.py', test_rows[:20])
        
        baseline_train_metrics = calculate_metrics(baseline_train_results)
        baseline_test_metrics = calculate_metrics(baseline_test_results)
        
        display_metrics(baseline_train_metrics, baseline_test_metrics, f"Baseline Summary (Store: {store_ids[0]})")
    
    # =========================================================================
    # PROCESS EACH STORE
    # =========================================================================
    
    print("\n" + "=" * 120)
    print(f"🏪 PROCESSING {len(store_ids)} STORE(S)")
    print("=" * 120)
    
    all_store_results = []
    
    for idx, store_id in enumerate(store_ids, 1):
        print(f"\n{'─' * 80}")
        print(f"   [{idx}/{len(store_ids)}] STORE: {store_id}")
        print(f"{'─' * 80}")
        
        store_rows = filter_by_store(all_rows, store_id)
        
        result = process_single_store(
            store_id=store_id,
            store_rows=store_rows,
            models_to_test=models_to_test,
            verbose=True
        )
        
        all_store_results.append(result)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 120)
    print("📊 FINAL RESULTS - ALL STORES")
    print("=" * 120)
    
    display_store_summary(all_store_results)
    
    # List generated scripts
    successful = [r for r in all_store_results if r is not None and r.get('status') == 'success']
    if successful:
        print(f"\n   📁 Generated Scripts:")
        for r in successful:
            print(f"      - {r['final_script']}")
    
    print("\n" + "=" * 120 + "\n")


if __name__ == "__main__":
    main()

"""
improvement.py - Template B: Iterative Improvement Prompt

This module builds the prompt for IMPROVING an existing extraction script.
It's used when we have code that partially works but needs fixes.

=== KEY PRINCIPLE ===

Only TRAINING set failures are shown to the model!
The test set is shown for reference only - never used for feedback.
This prevents overfitting to the test set.

=== PROMPT STRUCTURE ===

Section 1: PERFORMANCE SUMMARY
    - Training set metrics (what we're optimizing)
    - Test set metrics (for reference only)
    - Latency statistics

Section 2: CURRENT CODE
    - The existing script that needs improvement
    - Wrapped in code block for clarity

Section 3: TRAINING SET RESULTS
    - What's working (correct counts)
    - Sets expectations for what to maintain

Section 4: TRAINING FAILURES
    - Specific order ID failures with expected vs got
    - Specific subtotal failures with expected vs got
    - HTML context for failed cases

Section 5: FIX REQUIREMENTS
    - Instructions for what to fix
    - Latency optimization guidance (secondary priority)

=== USAGE ===

    from prompt_builders import improvement
    
    prompt = improvement.build_prompt(
        script_path='candidate_script.py',
        train_results=train_results,
        test_results=test_results,
        train_rows=train_rows,
        iteration=1
    )
    response = call_llm(prompt)
"""

from .base import extract_combined_context, CONTEXT_WINDOW_SIZE


# =============================================================================
# CONFIGURATION
# =============================================================================

# Target pass rate (%) - used in prompt to show the goal
MIN_PASS_RATE = 70

# Maximum number of failure examples to show in detail
MAX_FAILURE_EXAMPLES = 5


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(script_path, train_results, test_results, train_rows, iteration):
    """
    Build the prompt for script improvement (Template B).
    
    IMPORTANT: Only TRAINING failures are shown to the model.
    Test set results are shown for reference but NOT used for feedback.
    This ensures the model doesn't overfit to the test set.
    
    Args:
        script_path: Path to the current candidate script to read
        train_results: Test results on training set (used for feedback)
                      List of dicts with keys: row_id, success, order_id_correct,
                      subtotal_correct, expected_order_id, expected_subtotal,
                      extracted_order_id, extracted_subtotal, time
        test_results: Test results on test set (for reference only)
                     Same format as train_results
        train_rows: Original training data for context extraction
                   List of dicts with keys: row_id, html_content, order_id, subtotal
        iteration: Current improvement iteration number (1-5)
    
    Returns:
        Complete prompt string ready to send to the LLM
    
    Example:
        >>> prompt = build_prompt('script.py', train_results, test_results, train_rows, 1)
        >>> print(f"Prompt size: {len(prompt)} chars")
    """
    
    # Read the current script
    with open(script_path, 'r') as f:
        current_script = f.read()
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    
    train_total = len(train_results)
    train_oid_correct = sum(1 for r in train_results if r['order_id_correct'])
    train_sub_correct = sum(1 for r in train_results if r['subtotal_correct'])
    train_both_correct = sum(1 for r in train_results if r['success'])
    
    test_total = len(test_results)
    test_oid_correct = sum(1 for r in test_results if r['order_id_correct'])
    test_sub_correct = sum(1 for r in test_results if r['subtotal_correct'])
    test_both_correct = sum(1 for r in test_results if r['success'])
    
    # Calculate latency stats
    train_latencies = [r.get('time', 0) for r in train_results]
    test_latencies = [r.get('time', 0) for r in test_results]
    train_avg_latency = sum(train_latencies) / len(train_latencies) if train_latencies else 0
    test_avg_latency = sum(test_latencies) / len(test_latencies) if test_latencies else 0
    
    # =========================================================================
    # SECTION 1: PERFORMANCE SUMMARY
    # =========================================================================
    
    prompt = f"""
================================================================================
ITERATION {iteration}: SCRIPT IMPROVEMENT NEEDED
================================================================================

=== SECTION 1: PERFORMANCE SUMMARY ===

Training Set Performance:
- Order ID Correct: {train_oid_correct}/{train_total} ({train_oid_correct * 100 // train_total if train_total else 0}%)
- Subtotal Correct: {train_sub_correct}/{train_total} ({train_sub_correct * 100 // train_total if train_total else 0}%)
- Both Correct: {train_both_correct}/{train_total} ({train_both_correct * 100 // train_total if train_total else 0}%)
- Avg Latency: {train_avg_latency:.1f}ms

Holdout Test Set Performance (for reference only):
- Order ID Correct: {test_oid_correct}/{test_total} ({test_oid_correct * 100 // test_total if test_total else 0}%)
- Subtotal Correct: {test_sub_correct}/{test_total} ({test_sub_correct * 100 // test_total if test_total else 0}%)
- Both Correct: {test_both_correct}/{test_total} ({test_both_correct * 100 // test_total if test_total else 0}%)
- Avg Latency: {test_avg_latency:.1f}ms
- Target: {MIN_PASS_RATE}%

"""
    
    # =========================================================================
    # SECTION 2: CURRENT CODE
    # =========================================================================
    
    prompt += f"""=== SECTION 2: CURRENT CODE (needs improvement) ===

```python
{current_script}
```

"""
    
    # =========================================================================
    # SECTION 3: TRAINING SET RESULTS
    # =========================================================================
    
    prompt += "=== SECTION 3: TRAINING SET RESULTS ===\n\n"
    
    oid_working = [r for r in train_results if r['order_id_correct']]
    sub_working = [r for r in train_results if r['subtotal_correct']]
    
    prompt += f"✅ Order IDs correct: {len(oid_working)}/{train_total}\n"
    prompt += f"✅ Subtotals correct: {len(sub_working)}/{train_total}\n\n"
    
    # =========================================================================
    # SECTION 4: TRAINING FAILURES (with HTML context)
    # =========================================================================
    
    prompt += "=== SECTION 4: TRAINING FAILURES (with HTML context) ===\n\n"
    
    # Show order_id failures (just the expected vs got, no heavy context)
    oid_failed = [r for r in train_results if not r['order_id_correct']]
    if oid_failed:
        prompt += f"❌ ORDER ID FAILURES ({len(oid_failed)}/{train_total}):\n"
        for r in oid_failed[:MAX_FAILURE_EXAMPLES]:
            expected = r['expected_order_id']
            got = r.get('extracted_order_id') or 'NONE'
            prompt += f"  row_{r['row_id']}: Expected \"{expected}\" | Got \"{got}\"\n"
        if len(oid_failed) > MAX_FAILURE_EXAMPLES:
            prompt += f"  ... and {len(oid_failed) - MAX_FAILURE_EXAMPLES} more\n"
        prompt += "\n"
    
    # Show subtotal failures with HTML context to help debug
    sub_failed = [r for r in train_results if not r['subtotal_correct']]
    if sub_failed:
        prompt += f"❌ SUBTOTAL FAILURES ({len(sub_failed)}/{train_total}):\n"
        for r in sub_failed[:MAX_FAILURE_EXAMPLES]:
            expected = r['expected_subtotal']
            got = r.get('extracted_subtotal') or 'NONE'
            prompt += f"  row_{r['row_id']}: Expected \"{expected}\" | Got \"{got}\"\n"
            
            # Add HTML context for subtotal failures to help debug
            row_data = next((row for row in train_rows if row['row_id'] == r['row_id']), None)
            if row_data:
                ctx_info = extract_combined_context(
                    row_data['html_content'], 
                    r['expected_order_id'], 
                    r['expected_subtotal'], 
                    CONTEXT_WINDOW_SIZE
                )
                # FIX: Use the best available context, not just order_id_context
                context = None
                if ctx_info.get('type') == 'combined' and ctx_info.get('context'):
                    context = ctx_info['context'][:2000]
                elif ctx_info.get('subtotal_context'):
                    context = ctx_info['subtotal_context'][:2000]
                elif ctx_info.get('order_id_context'):
                    context = ctx_info['order_id_context'][:2000]
                elif ctx_info.get('context'):
                    context = ctx_info['context'][:2000]
                
                if context:
                    prompt += f"    HTML Context:\n{context}\n\n"
        
        if len(sub_failed) > MAX_FAILURE_EXAMPLES:
            prompt += f"  ... and {len(sub_failed) - MAX_FAILURE_EXAMPLES} more\n"
    
    # =========================================================================
    # SECTION 5: FIX REQUIREMENTS
    # =========================================================================
    
    prompt += """
================================================================================
FIX REQUIREMENTS
================================================================================

1. Fix the extraction logic to handle the failing patterns
2. Maintain correctness for cases that already work
3. Order IDs: Look for "Order #" followed by digits in visible HTML elements
4. Subtotals: Look for price patterns near "Total", "Subtotal" text
5. Keep the same output format: [{"order_id": "...", "subtotal": "..."}]
6. CRITICAL: Return partial results! If order_id is found but subtotal is not, still return the order_id with empty subtotal
7. DO NOT require both fields - return whatever you can extract

LATENCY OPTIMIZATION (secondary priority - NEVER compromise accuracy):
- Use efficient parsing: avoid unnecessary DOM traversals
- Use compiled regex patterns (re.compile) for patterns used multiple times
- Exit early when values are found - don't keep searching unnecessarily
- Avoid parsing the entire HTML multiple times
- Use specific selectors over generic find_all when possible
- BUT: Never sacrifice accuracy for speed - correctness is always #1 priority

Write the IMPROVED Python script:
"""
    
    return prompt

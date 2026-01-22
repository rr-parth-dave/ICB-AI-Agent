"""
first_inference.py - Template A: Initial Code Generation Prompt

This module builds the prompt for the FIRST attempt at generating extraction code.
It's used when we have no existing code and need the LLM to create one from scratch.

=== PROMPT STRUCTURE ===

Section 1: OUTPUT FORMAT
    - What the script should produce (JSON format)
    - Command-line interface specification
    - Rules for handling edge cases

Section 2: GROUND TRUTH VALUES
    - All expected order IDs listed
    - All expected subtotals listed
    - Helps LLM understand the pattern of values

Section 3: HTML CONTEXT WINDOWS
    - Smart context snippets around each ground truth value
    - Combined contexts when values are close together
    - Separate contexts when values are far apart

Section 4: INSTRUCTIONS
    - Hints for finding order IDs
    - Hints for finding subtotals
    - Best practices for HTML parsing

=== USAGE ===

    from prompt_builders import first_inference
    
    prompt = first_inference.build_prompt(train_rows)
    response = call_llm(prompt)
"""

import json
from .base import extract_combined_context, CONTEXT_WINDOW_SIZE


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum number of training examples to include in the prompt
# More examples = better pattern recognition, but larger prompt
MAX_TRAINING_EXAMPLES = 10


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(train_rows):
    """
    Build the prompt for initial code generation (Template A).
    
    This prompt teaches the LLM how to write an extraction script by showing:
    1. The expected output format
    2. All the ground truth values it needs to extract
    3. HTML context around each ground truth value
    4. Tips for finding order IDs and subtotals
    
    Args:
        train_rows: List of training data dicts with keys:
                   - row_id: Unique identifier
                   - html_content: Full HTML of the page
                   - order_id: Ground truth order ID
                   - subtotal: Ground truth subtotal
    
    Returns:
        Complete prompt string ready to send to the LLM
    
    Example:
        >>> prompt = build_prompt(train_rows[:10])
        >>> print(f"Prompt size: {len(prompt)} chars")
    """
    
    # =========================================================================
    # SECTION 1: OUTPUT FORMAT
    # =========================================================================
    
    prompt = """
================================================================================
TASK: Write a Python script to extract Order Number and Subtotal from HTML
================================================================================

=== SECTION 1: OUTPUT FORMAT ===

INPUT: Python script receives one argument: path to HTML file
OUTPUT: Print ONLY valid JSON to STDOUT

Format:
[{"order_id": "<exact_order_number>", "subtotal": "<exact_amount>"}]

Example output:
[{"order_id": "902002760431352", "subtotal": "149.03"}]

Rules:
- order_id: Exact value as it appears (no extra prefixes, no whitespace)
- subtotal: Exact numeric value (no "$", no "Total:", no whitespace)
- Always return a list (even for single order)
- If no order found, print []
- Use BeautifulSoup for HTML parsing
- IMPORTANT: Return order_id even if subtotal is not found (use empty string "")
- IMPORTANT: Return subtotal even if order_id is not found (use empty string "")
- DO NOT require both fields to be present - return partial results!

"""
    
    # =========================================================================
    # SECTION 2: GROUND TRUTH VALUES
    # =========================================================================
    
    prompt += "=== SECTION 2: GROUND TRUTH VALUES ===\n\n"
    prompt += "These are the EXACT values the script must extract:\n\n"
    
    examples_to_show = train_rows[:MAX_TRAINING_EXAMPLES]
    
    # List all order IDs
    prompt += f"Order IDs to extract ({len(examples_to_show)} examples shown):\n"
    for i, row in enumerate(examples_to_show, 1):
        prompt += f"  {i}. \"{row['order_id']}\"\n"
    
    # List all subtotals
    prompt += f"\nSubtotals to extract ({len(examples_to_show)} examples shown):\n"
    for i, row in enumerate(examples_to_show, 1):
        prompt += f"  {i}. \"{row['subtotal']}\"\n"
    
    prompt += "\n"
    
    # =========================================================================
    # SECTION 3: HTML CONTEXT WINDOWS
    # =========================================================================
    
    prompt += "=== SECTION 3: HTML CONTEXT WINDOWS ===\n\n"
    prompt += f"Each example shows ~{CONTEXT_WINDOW_SIZE} chars before and after the target values.\n"
    prompt += "When order_id and subtotal are close, they're shown in one combined context.\n"
    prompt += "Study these patterns carefully to understand the HTML structure.\n\n"
    
    for idx, row in enumerate(examples_to_show, 1):
        html = row['html_content']
        order_id = row['order_id']
        subtotal = row['subtotal']
        
        # Extract smart context for this example
        ctx_info = extract_combined_context(html, order_id, subtotal, CONTEXT_WINDOW_SIZE)
        
        # Header for this example
        prompt += f"--- EXAMPLE {idx} (row_{row['row_id']}) ---\n"
        prompt += f"Ground Truth: order_id=\"{order_id}\", subtotal=\"{subtotal}\"\n"
        prompt += f"Expected Output: [{json.dumps({'order_id': order_id, 'subtotal': subtotal})}]\n\n"
        
        # Context based on type
        if ctx_info['type'] == 'combined':
            prompt += f"[COMBINED CONTEXT - values are {ctx_info['distance']} chars apart]\n"
            prompt += f"{ctx_info['context']}\n"
            prompt += "[/COMBINED CONTEXT]\n\n"
        elif ctx_info['type'] == 'separate':
            prompt += f"[ORDER_ID CONTEXT]\n{ctx_info['order_id_context']}\n[/ORDER_ID CONTEXT]\n\n"
            prompt += f"[SUBTOTAL CONTEXT]\n{ctx_info['subtotal_context']}\n[/SUBTOTAL CONTEXT]\n\n"
        elif ctx_info['type'] == 'order_id_only':
            prompt += f"[ORDER_ID CONTEXT]\n{ctx_info['order_id_context']}\n[/ORDER_ID CONTEXT]\n"
            prompt += f"[SUBTOTAL - NOT FOUND IN HTML - look for patterns with 'Total', 'Subtotal', or price amounts]\n\n"
        elif ctx_info['type'] == 'subtotal_only':
            prompt += f"[ORDER_ID - NOT FOUND IN HTML - look for patterns with 'Order #', 'Order ID', or numeric IDs]\n"
            prompt += f"[SUBTOTAL CONTEXT]\n{ctx_info['subtotal_context']}\n[/SUBTOTAL CONTEXT]\n\n"
        else:
            prompt += f"[FALLBACK CONTEXT - neither value found directly, study HTML structure]\n"
            prompt += f"{ctx_info['context']}\n"
            prompt += "[/FALLBACK CONTEXT]\n\n"
        
        prompt += "\n"
    
    # =========================================================================
    # SECTION 4: INSTRUCTIONS
    # =========================================================================
    
    prompt += """
================================================================================
INSTRUCTIONS
================================================================================

1. Study the HTML patterns above carefully
2. Order IDs often appear:
   - Near "Order #" or "Order ID" text
   - In elements with class names containing "order"
   - As long numeric strings (e.g., 15-digit numbers)
3. Subtotals often appear:
   - Near "Total", "Subtotal", "Grand Total" text
   - In elements with class names containing "total"
   - As decimal numbers (e.g., "149.03")
4. Use BeautifulSoup to parse and navigate the HTML
5. Use regex patterns to extract values when needed
6. Handle cases where values might not be found

Now write the complete Python script:
"""
    
    return prompt


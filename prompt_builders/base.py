"""
base.py - Shared Utilities for Context Extraction

This module provides the foundational utilities used by both prompt templates:
- String normalization for comparison
- Value position finding in HTML
- Smart context window extraction with coalescing

These utilities enable "smart context" - extracting focused snippets of HTML
around ground truth values instead of sending entire HTML files.
"""

import re


# =============================================================================
# CONFIGURATION
# =============================================================================

# Characters to extract before/after ground truth for context windows
# This creates a focused view of the HTML around the target values
CONTEXT_WINDOW_SIZE = 2000


# =============================================================================
# STRING UTILITIES
# =============================================================================

def normalize_string(s):
    """
    Normalize a string for comparison (removes formatting differences).
    
    Removes: dollar signs, spaces, commas
    Converts to lowercase for case-insensitive comparison.
    
    Args:
        s: String to normalize (or None)
    
    Returns:
        Normalized string (empty string if input was None)
    
    Examples:
        >>> normalize_string("$149.03")
        '149.03'
        >>> normalize_string("Order #123")
        'order#123'
    """
    if s is None:
        return ''
    return str(s).replace('$', '').replace(' ', '').replace(',', '').lower().strip()


# =============================================================================
# CONTEXT EXTRACTION
# =============================================================================

def find_value_position(html, value):
    """
    Find the position of a ground truth value in HTML.
    
    Tries multiple patterns to locate the value:
    1. Direct string match
    2. Common variations (e.g., "Order #123", ">123<", "$45.00")
    3. Regex patterns for price amounts
    
    Args:
        html: HTML string to search in
        value: Value to find
    
    Returns:
        Tuple of (start_position, end_position) or (None, None) if not found
    
    Examples:
        >>> html = '<span>Order #12345</span>'
        >>> find_value_position(html, '12345')
        (14, 19)
    """
    if not value:
        return None, None
    
    # Try direct match first (most common case)
    pos = html.find(value)
    if pos != -1:
        return pos, pos + len(value)
    
    # Try common variations for order IDs and prices
    variations = [
        f'Order #{value}',
        f'Order # {value}',
        f'Order#{value}',
        f'>{value}<',
        f'${value}',
        f'Total: ${value}',
        f'Total:${value}',
        f'Total: {value}',
    ]
    for v in variations:
        pos = html.find(v)
        if pos != -1:
            val_pos = v.find(value)
            return pos + val_pos, pos + val_pos + len(value)
    
    # Try regex for decimal values (prices)
    if '.' in value:
        patterns = [
            re.escape(value),
            r'\$\s*' + re.escape(value),
            re.escape(value.replace('.', ',')),
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                full_match = match.group()
                val_pos = full_match.find(value) if value in full_match else 0
                return match.start() + val_pos, match.start() + val_pos + len(value)
    
    return None, None


def extract_combined_context(html, order_id, subtotal, window_size=None):
    """
    Extract context windows around ground truth values.
    
    Smart Coalescing:
        If order_id and subtotal are close together (within window_size),
        combines them into a single context to reduce prompt size and
        show their relationship in the HTML structure.
    
    Args:
        html: Full HTML content
        order_id: Ground truth order ID to find
        subtotal: Ground truth subtotal to find
        window_size: Characters to include before/after each value
                    (defaults to CONTEXT_WINDOW_SIZE)
    
    Returns:
        Dict with context information:
        - type: 'combined', 'separate', 'order_id_only', 'subtotal_only', or 'fallback'
        - order_id, subtotal: The values being searched for
        - context/order_id_context/subtotal_context: Extracted HTML snippets
        - order_id_found, subtotal_found: Whether each value was located
        - distance: (only for 'combined') Characters between the two values
    
    Example:
        >>> ctx = extract_combined_context(html, '12345', '99.99')
        >>> if ctx['type'] == 'combined':
        ...     print(f"Values are {ctx['distance']} chars apart")
    """
    if window_size is None:
        window_size = CONTEXT_WINDOW_SIZE
    
    oid_start, oid_end = find_value_position(html, order_id)
    sub_start, sub_end = find_value_position(html, subtotal)
    
    # Case 1: Both found and close together - combine into one context
    if oid_start is not None and sub_start is not None:
        distance = abs(oid_start - sub_start)
        if distance <= window_size:
            combined_start = max(0, min(oid_start, sub_start) - window_size)
            combined_end = min(len(html), max(oid_end, sub_end) + window_size)
            return {
                'type': 'combined',
                'order_id': order_id,
                'subtotal': subtotal,
                'context': html[combined_start:combined_end],
                'distance': distance,
                'order_id_found': True,
                'subtotal_found': True
            }
        else:
            # Both found but far apart - separate contexts
            oid_ctx_start = max(0, oid_start - window_size)
            oid_ctx_end = min(len(html), oid_end + window_size)
            sub_ctx_start = max(0, sub_start - window_size)
            sub_ctx_end = min(len(html), sub_end + window_size)
            return {
                'type': 'separate',
                'order_id': order_id,
                'order_id_context': html[oid_ctx_start:oid_ctx_end],
                'subtotal': subtotal,
                'subtotal_context': html[sub_ctx_start:sub_ctx_end],
                'order_id_found': True,
                'subtotal_found': True
            }
    
    # Case 2: Only order_id found
    if oid_start is not None:
        oid_ctx_start = max(0, oid_start - window_size)
        oid_ctx_end = min(len(html), oid_end + window_size)
        return {
            'type': 'order_id_only',
            'order_id': order_id,
            'order_id_context': html[oid_ctx_start:oid_ctx_end],
            'subtotal': subtotal,
            'subtotal_context': None,
            'order_id_found': True,
            'subtotal_found': False
        }
    
    # Case 3: Only subtotal found
    if sub_start is not None:
        sub_ctx_start = max(0, sub_start - window_size)
        sub_ctx_end = min(len(html), sub_end + window_size)
        return {
            'type': 'subtotal_only',
            'order_id': order_id,
            'order_id_context': None,
            'subtotal': subtotal,
            'subtotal_context': html[sub_ctx_start:sub_ctx_end],
            'order_id_found': False,
            'subtotal_found': True
        }
    
    # Case 4: Neither found - return beginning of HTML as fallback
    return {
        'type': 'fallback',
        'order_id': order_id,
        'subtotal': subtotal,
        'context': html[:min(len(html), window_size * 2)],
        'order_id_found': False,
        'subtotal_found': False
    }


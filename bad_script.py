#!/usr/bin/env python3
"""
bad_script.py - Intentionally Limited Baseline Extraction Script

Purpose:
    This script serves as a BASELINE for the Auto-Heal Pipeline. It uses
    naive extraction patterns that are intentionally limited/broken to
    establish a starting point for comparison.

Why it exists:
    - Demonstrates 0% accuracy on complex HTML (like Target's OCP pages)
    - Shows the improvement achieved by AI-generated scripts
    - Provides a control group for measuring pipeline effectiveness

Limitations (by design):
    - Uses simple class lookups that don't match Target's HTML structure
    - Falls back to naive regex patterns that often fail
    - Only returns results if BOTH order_id AND subtotal are found
    - Cannot handle the complex, deeply nested Target HTML

Usage:
    python bad_script.py <path_to_html_file>

Output:
    JSON array: [{"order_id": "...", "subtotal": "..."}]
    Empty array [] if extraction fails
"""

import sys
import json
import re
from bs4 import BeautifulSoup


def extract_order_id(soup, html):
    """
    Attempt to extract order ID using naive patterns.
    
    Args:
        soup: BeautifulSoup parsed HTML object
        html: Raw HTML string for regex fallback
    
    Returns:
        Order ID string or None if not found
    """
    order_id = None
    
    # Method 1: Simple class lookup (won't work on Target HTML)
    pkg_id = soup.find('span', class_='pkg-id')
    if pkg_id:
        order_text = pkg_id.get_text(strip=True)
        order_id = order_text.replace('Order #', '').replace('Order#', '').strip()
    
    # Method 2: Naive regex fallback
    if not order_id:
        match = re.search(r'Order\s*#\s*(\d+)', html)
        if match:
            order_id = match.group(1)
    
    return order_id


def extract_subtotal(soup, html):
    """
    Attempt to extract subtotal using naive patterns.
    
    Args:
        soup: BeautifulSoup parsed HTML object
        html: Raw HTML string for regex fallback
    
    Returns:
        Subtotal string or None if not found
    """
    subtotal = None
    
    # Method 1: Simple class lookup (won't work on Target HTML)
    grand_total = soup.find('span', class_='grand-total')
    if grand_total:
        total_text = grand_total.get_text(strip=True)
        subtotal = total_text.replace('Total:', '').replace('$', '').strip()
    
    # Method 2: Naive regex fallback (just finds first dollar amount)
    if not subtotal:
        match = re.search(r'\$(\d+\.\d{2})', html)
        if match:
            subtotal = match.group(1)
    
    return subtotal


def main():
    """
    Main entry point for the baseline extraction script.
    
    Reads HTML file from command line argument and attempts extraction
    using intentionally limited patterns.
    """
    # Validate command line arguments
    if len(sys.argv) != 2:
        print("[]")
        return
    
    # Read HTML file
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            html = f.read()
    except Exception:
        print("[]")
        return
    
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract values using naive patterns
    order_id = extract_order_id(soup, html)
    subtotal = extract_subtotal(soup, html)
    
    # Output results (only if BOTH are found - another intentional limitation)
    if order_id and subtotal:
        result = [{"order_id": order_id, "subtotal": subtotal}]
        print(json.dumps(result))
    else:
        print("[]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
script_utils.py - Utility functions for script management

This module provides utilities for:
1. Checking ground truth presence in HTML
2. Cleaning up candidate scripts
3. Renaming the final best script

Usage:
    from script_utils import check_ground_truth_presence, cleanup_and_save_final
"""

import os
import glob
import re
import shutil


# =============================================================================
# GROUND TRUTH PRESENCE CHECK
# =============================================================================

def normalize_for_search(value):
    """
    Normalize a value for searching in HTML.
    Removes common formatting differences but preserves the core value.
    """
    if not value:
        return ""
    s = str(value).strip()
    # Remove currency symbols and common formatting
    s = re.sub(r'[$€£¥]', '', s)
    s = re.sub(r'\s+', '', s)  # Remove whitespace
    return s.lower()


def value_exists_in_html(html, value):
    """
    Check if a ground truth value exists anywhere in the HTML.
    Uses multiple matching strategies to account for formatting.
    
    Args:
        html: The HTML content to search in
        value: The ground truth value to find
    
    Returns:
        True if value is found, False otherwise
    """
    if not html or not value:
        return False
    
    value_str = str(value).strip()
    if not value_str:
        return False
    
    html_lower = html.lower()
    
    # Strategy 1: Exact match
    if value_str.lower() in html_lower:
        return True
    
    # Strategy 2: Normalized match (no spaces, no currency symbols)
    normalized_val = normalize_for_search(value_str)
    normalized_html = normalize_for_search(html)
    if normalized_val and normalized_val in normalized_html:
        return True
    
    # Strategy 3: For prices, try without $ sign
    if value_str.startswith('$'):
        without_dollar = value_str[1:].strip()
        if without_dollar.lower() in html_lower:
            return True
    
    # Strategy 4: For order IDs with #, try without
    if '#' in value_str:
        without_hash = value_str.replace('#', '').strip()
        if without_hash.lower() in html_lower:
            return True
    
    return False


def check_ground_truth_presence(rows, set_name="Data"):
    """
    Check how often ground truth values appear in the HTML for a set of rows.
    
    Args:
        rows: List of data rows with 'html_content', 'order_id', 'subtotal'
        set_name: Name of the set for display (e.g., "Training", "Test")
    
    Returns:
        Dict with presence statistics
    """
    total = len(rows)
    if total == 0:
        return {'order_id_present': 0, 'subtotal_present': 0, 'both_present': 0,
                'order_id_pct': 0, 'subtotal_pct': 0, 'both_pct': 0}
    
    order_id_found = 0
    subtotal_found = 0
    both_found = 0
    
    for row in rows:
        html = row.get('html_content', '')
        oid = row.get('order_id', '')
        sub = row.get('subtotal', '')
        
        oid_present = value_exists_in_html(html, oid)
        sub_present = value_exists_in_html(html, sub)
        
        if oid_present:
            order_id_found += 1
        if sub_present:
            subtotal_found += 1
        if oid_present and sub_present:
            both_found += 1
    
    return {
        'set_name': set_name,
        'total': total,
        'order_id_present': order_id_found,
        'subtotal_present': subtotal_found,
        'both_present': both_found,
        'order_id_pct': (order_id_found * 100) // total if total else 0,
        'subtotal_pct': (subtotal_found * 100) // total if total else 0,
        'both_pct': (both_found * 100) // total if total else 0
    }


def display_ground_truth_presence(train_stats, test_stats):
    """
    Display a formatted table of ground truth presence statistics.
    
    Args:
        train_stats: Stats dict from check_ground_truth_presence for training set
        test_stats: Stats dict from check_ground_truth_presence for test set
    """
    print("\n   📋 Ground Truth Presence in HTML (can the value even be found?):")
    print(f"   ┌────────────────────┬──────────────────┬──────────────────┐")
    print(f"   │ Ground Truth       │ Training Set     │ Test Set         │")
    print(f"   ├────────────────────┼──────────────────┼──────────────────┤")
    print(f"   │ Order ID Present   │ {train_stats['order_id_present']:>3}/{train_stats['total']:<3} ({train_stats['order_id_pct']:>3}%)   │ {test_stats['order_id_present']:>3}/{test_stats['total']:<3} ({test_stats['order_id_pct']:>3}%)   │")
    print(f"   │ Subtotal Present   │ {train_stats['subtotal_present']:>3}/{train_stats['total']:<3} ({train_stats['subtotal_pct']:>3}%)   │ {test_stats['subtotal_present']:>3}/{test_stats['total']:<3} ({test_stats['subtotal_pct']:>3}%)   │")
    print(f"   │ Both Present       │ {train_stats['both_present']:>3}/{train_stats['total']:<3} ({train_stats['both_pct']:>3}%)   │ {test_stats['both_present']:>3}/{test_stats['total']:<3} ({test_stats['both_pct']:>3}%)   │")
    print(f"   └────────────────────┴──────────────────┴──────────────────┘")
    
    # Warning if subtotal presence is low
    if train_stats['subtotal_pct'] < 50 or test_stats['subtotal_pct'] < 50:
        print(f"\n   ⚠️  WARNING: Subtotal ground truth is often NOT present in HTML!")
        print(f"       This limits the maximum achievable subtotal accuracy.")


# =============================================================================
# SCRIPT CLEANUP AND FINAL SAVE
# =============================================================================

def cleanup_and_save_final(best_script_path, store_id, model_type, iteration_count, keep_best=True):
    """
    Clean up candidate scripts for a store and save the best one with a descriptive name.
    
    Args:
        best_script_path: Path to the best performing script
        store_id: The store identifier (included in output filename)
        model_type: The model that generated it (e.g., "claude_fast")
        iteration_count: Number of improvement iterations performed
        keep_best: If True, keeps a copy as the final script
    
    Returns:
        Path to the final saved script
    """
    # Generate final script name with store_id
    # Sanitize store_id to be safe for filenames
    safe_store_id = str(store_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
    final_name = f"final_{safe_store_id}_{model_type}_iter{iteration_count}.py"
    
    # Find all candidate scripts for this store
    candidate_patterns = [
        f"candidate_script_{safe_store_id}_*.py",
        "candidate_script_*.py",  # Also clean up any orphaned scripts
        "fixed_script.py"
    ]
    
    all_candidates = []
    for pattern in candidate_patterns:
        all_candidates.extend(glob.glob(pattern))
    
    # Remove duplicates
    all_candidates = list(set(all_candidates))
    
    # Copy best script to final name first (before deleting)
    if keep_best and os.path.exists(best_script_path):
        shutil.copy(best_script_path, final_name)
        # Print is handled by caller for cleaner output in multi-store mode
    
    # Delete all candidate scripts (except the final one)
    deleted_count = 0
    for script in all_candidates:
        if script != final_name and os.path.exists(script):
            try:
                os.remove(script)
                deleted_count += 1
            except Exception as e:
                pass  # Silent in multi-store mode
    
    return final_name


def get_candidate_scripts():
    """
    Get a list of all candidate script files in the current directory.
    
    Returns:
        List of candidate script paths
    """
    patterns = [
        "candidate_script_*.py",
        "fixed_script.py"
    ]
    
    scripts = []
    for pattern in patterns:
        scripts.extend(glob.glob(pattern))
    
    return list(set(scripts))


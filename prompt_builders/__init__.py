"""
prompt_builders - Modular Prompt Building for Auto-Heal Pipeline

This package contains the prompt templates used to communicate with LLMs:

Modules:
    base            - Shared utilities for context extraction and normalization
    first_inference - Template A: Initial code generation prompt
    improvement     - Template B: Iterative improvement prompt

Usage:
    from prompt_builders import first_inference, improvement
    from prompt_builders.base import CONTEXT_WINDOW_SIZE
    
    # Build initial prompt
    prompt = first_inference.build_prompt(train_rows)
    
    # Build improvement prompt
    prompt = improvement.build_prompt(script_path, train_results, test_results, train_rows, iteration)
"""

from . import first_inference
from . import improvement
from . import base

__all__ = ['first_inference', 'improvement', 'base']


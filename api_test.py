#!/usr/bin/env python3
"""
api_test.py - LLM API Client for Multiple Providers

This module provides a unified interface to call LLMs from three major providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)

Features:
- Supports both "capable" (smart) and "fast" (cheap) model variants
- Tracks response times for performance comparison
- Provides individual and unified API callers
- Includes a test suite to verify all endpoints

Usage:
    # Import and use individual callers
    from api_test import call_claude, call_openai, call_gemini
    
    response = call_claude("Write a hello world in Python")
    
    # Or use the unified caller
    from api_test import call_llm
    
    response = call_llm("Write hello world", model_type="openai")
    
    # Run tests directly
    python api_test.py

Configuration:
    Change USE_MODEL to switch default behavior:
    - "claude"  : Use Claude Opus (capable)
    - "openai"  : Use GPT-5.1 (capable)
    - "gemini"  : Use Gemini 3 Pro (capable)
    - "all"     : Use all three capable models
    - "fast"    : Use all three fast models
"""

import requests
import json
import time
import os

# =============================================================================
# SSL CERTIFICATE CONFIGURATION
# =============================================================================
# For corporate environments with SSL inspection (e.g., Zscaler)
# The corporate cert must be COMBINED with the default CA bundle

def _get_combined_cert_bundle():
    """
    Create a combined certificate bundle that includes both:
    1. Default system/certifi CA certificates
    2. Corporate certificates (Rakuten/Zscaler)
    
    This is necessary because corporate SSL inspection requires the 
    corporate root CA, but we still need standard CAs for the chain.
    """
    import certifi
    import tempfile
    
    # Find corporate cert
    corp_cert_path = os.environ.get('SSL_CERT_FILE') or os.path.expanduser('~/rakuten-ca.pem')
    
    if not os.path.exists(corp_cert_path):
        return True  # No corporate cert, use defaults
    
    # Combine default + corporate certs
    try:
        with open(certifi.where()) as f:
            default_certs = f.read()
        with open(corp_cert_path) as f:
            corp_certs = f.read()
        
        combined_path = '/tmp/combined_ca_bundle.pem'
        with open(combined_path, 'w') as f:
            f.write(default_certs + '\n' + corp_certs)
        
        return combined_path
    except Exception:
        return True  # Fall back to defaults on any error

SSL_CERT_PATH = _get_combined_cert_bundle()

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Key for Rakuten AI Platform
API_KEY = "raik-pat-3c78ab7bbr10aibf8dc780cde647d031b2d8c9673c7244bf8dc780cde647d031"

# Simple test prompt for connectivity checks
TEST_PROMPT = "Hello! Just say 'Online'."

# Default model selection (change this to switch behavior)
# Options: "claude", "openai", "gemini", "all", "fast"
USE_MODEL = "all"

# =============================================================================
# MODEL CONFIGURATION - State of the Art (December 2024/2025)
# =============================================================================

# ANTHROPIC CLAUDE MODELS
CLAUDE_MODEL = "claude-opus-4-5-20251101"   # Claude Opus 4.5 - Most capable
CLAUDE_FAST = "claude-3-5-haiku-20241022"   # Claude 3.5 Haiku - Fastest/cheapest

# OPENAI MODELS
OPENAI_MODEL = "gpt-5.1"      # GPT-5.1 - Most capable
OPENAI_FAST = "gpt-4o-mini"   # GPT-4o Mini - Fastest/cheapest

# GOOGLE GEMINI MODELS
GEMINI_MODEL = "gemini-3-pro-preview"   # Gemini 3 Pro - Most capable
GEMINI_FAST = "gemini-3-pro-preview"    # Flash not available yet, fallback to Pro

# All models organized by provider and tier
ALL_MODELS = {
    "claude": {"capable": CLAUDE_MODEL, "fast": CLAUDE_FAST},
    "openai": {"capable": OPENAI_MODEL, "fast": OPENAI_FAST},
    "gemini": {"capable": GEMINI_MODEL, "fast": GEMINI_FAST}
}

# Track actual response times (populated during testing)
RESPONSE_TIMES = {}

# =============================================================================
# BASE API ENDPOINT
# =============================================================================

BASE_URL = "https://api.ai.public.rakuten-it.com"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_result(provider, model, status, time_taken, output=None):
    """
    Print formatted test result for a model.
    
    Args:
        provider: Provider name (e.g., "OpenAI", "Anthropic")
        model: Model name with optional tag
        status: HTTP status code or error string
        time_taken: Response time in seconds
        output: Optional response text to display
    """
    icon = "✅" if status == 200 else "❌"
    status_str = str(status) if isinstance(status, int) else status
    print(f"{provider:<12} | {model:<30} | {icon} {status_str:<6} | {time_taken:.2f}s")
    
    if output:
        # Clean and truncate output for display
        clean_output = output.replace('\n', ' ').strip()
        if len(clean_output) > 90:
            clean_output = clean_output[:90] + "..."
        print(f"             └── 💬 {clean_output}")
    
    print("-" * 90)


def make_request(url, headers, payload, timeout=30):
    """
    Make an HTTP POST request to an API endpoint.
    
    Args:
        url: Full URL to POST to
        headers: Request headers dict
        payload: JSON payload dict
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (status_code, response_data, duration_seconds)
    """
    try:
        start = time.time()
        # Use explicit cert path if available (for corporate SSL inspection)
        verify = SSL_CERT_PATH if SSL_CERT_PATH else True
        response = requests.post(url, headers=headers, json=payload, timeout=timeout, verify=verify)
        duration = time.time() - start
        
        # Parse JSON on success, raw text on failure
        data = response.json() if response.status_code == 200 else response.text
        return response.status_code, data, duration
    except Exception as e:
        return "ERR", str(e), 0


# =============================================================================
# TEST FUNCTIONS (for running API connectivity tests)
# =============================================================================

def test_openai():
    """Test all available OpenAI models on the platform."""
    url = f"{BASE_URL}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\n🤖 TESTING OPENAI (Endpoint: {BASE_URL}/openai/v1)")
    print("=" * 90)
    
    # Models to test: (model_name, display_tag)
    models = [
        ("gpt-5.1", "🧠CAPABLE"),
        ("gpt-4o-mini", "⚡FAST"),
        ("o4-mini", "⚡FAST"),
    ]
    
    for model, tag in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": TEST_PROMPT}]
        }
        status, data, duration = make_request(url, headers, payload)
        
        # Track response time for successful requests
        if status == 200:
            RESPONSE_TIMES[model] = duration
        
        # Display result
        if status == 200 and isinstance(data, dict):
            content = data.get('choices', [{}])[0].get('message', {}).get('content', 'Parse error')
            print_result("OpenAI", f"{model} {tag}", status, duration, content)
        else:
            error = data if isinstance(data, str) else str(data)
            print_result("OpenAI", model, status, duration, error)


def test_anthropic():
    """Test all available Claude models on the platform."""
    url = f"{BASE_URL}/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    print(f"\n🧠 TESTING ANTHROPIC (Endpoint: {BASE_URL}/anthropic/v1)")
    print("=" * 90)
    
    # Models to test: (model_name, display_tag)
    models = [
        (CLAUDE_MODEL, "🧠CAPABLE"),
        (CLAUDE_FAST, "⚡FAST"),
    ]
    
    for model, tag in models:
        payload = {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": TEST_PROMPT}]
        }
        status, data, duration = make_request(url, headers, payload)
        
        # Track response time for successful requests
        if status == 200:
            RESPONSE_TIMES[model] = duration
        
        # Display result
        if status == 200 and isinstance(data, dict):
            content = data.get('content', [{}])[0].get('text', 'Parse error')
            print_result("Anthropic", f"{model} {tag}", status, duration, content)
        else:
            error = data if isinstance(data, str) else str(data)
            print_result("Anthropic", model, status, duration, error)


def test_gemini():
    """Test all available Gemini models on the platform."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\n⚡ TESTING GEMINI (Endpoint: {BASE_URL}/google-vertexai/v1)")
    print("=" * 90)
    
    # Models to test (Flash not available yet, only Pro)
    models = [
        (GEMINI_MODEL, "🧠CAPABLE"),
    ]
    
    for model, tag in models:
        url = f"{BASE_URL}/google-vertexai/v1/publishers/google/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": TEST_PROMPT}]}],
            "generationConfig": {"thinkingConfig": {"includeThoughts": False}}
        }
        status, data, duration = make_request(url, headers, payload)
        
        # Track response time for successful requests
        if status == 200:
            RESPONSE_TIMES[model] = duration
        
        # Display result
        if status == 200 and isinstance(data, dict):
            text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Parse error')
            print_result("Google", f"{model} {tag}", status, duration, text)
        else:
            error = data if isinstance(data, str) else str(data)
            print_result("Google", model, status, duration, error)


# =============================================================================
# API CALLER FUNCTIONS (for use by other scripts)
# =============================================================================

def call_claude(prompt, max_tokens=4000, model=None, timeout=120):
    """
    Call Claude API and return the response text.
    
    Args:
        prompt: The prompt to send to Claude
        max_tokens: Maximum tokens in response (default: 4000)
        model: Model name (default: CLAUDE_MODEL)
        timeout: Request timeout in seconds (default: 120)
    
    Returns:
        Response text string
    
    Raises:
        Exception: If API call fails
    """
    if model is None:
        model = CLAUDE_MODEL
    
    url = f"{BASE_URL}/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    status, data, duration = make_request(url, headers, payload, timeout=timeout)
    
    if status == 200 and isinstance(data, dict):
        return data.get('content', [{}])[0].get('text', '')
    else:
        raise Exception(f"API call failed: {data}")


def call_openai(prompt, max_tokens=4000, model=None, timeout=120):
    """
    Call OpenAI API and return the response text.
    
    Args:
        prompt: The prompt to send to GPT
        max_tokens: Maximum tokens in response (default: 4000)
        model: Model name (default: OPENAI_MODEL)
        timeout: Request timeout in seconds (default: 120)
    
    Returns:
        Response text string
    
    Raises:
        Exception: If API call fails
    """
    if model is None:
        model = OPENAI_MODEL
    
    url = f"{BASE_URL}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    status, data, duration = make_request(url, headers, payload, timeout=timeout)
    
    if status == 200 and isinstance(data, dict):
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    else:
        raise Exception(f"API call failed: {data}")


def call_gemini(prompt, max_tokens=4000, model=None):
    """
    Call Gemini API and return the response text.
    
    Note: Gemini 3 Pro uses "thinking" tokens which count against the limit.
    We disable thinking mode for code generation to maximize output tokens.
    
    Args:
        prompt: The prompt to send to Gemini
        max_tokens: Maximum tokens in response (default: 4000)
        model: Model name (default: GEMINI_MODEL)
    
    Returns:
        Response text string
    
    Raises:
        Exception: If API call fails
    """
    if model is None:
        model = GEMINI_MODEL
    
    url = f"{BASE_URL}/google-vertexai/v1/publishers/google/models/{model}:generateContent"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Double max_tokens for Gemini 3 to account for thinking tokens
    effective_max_tokens = max_tokens * 2 if "gemini-3" in model.lower() else max_tokens
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": effective_max_tokens,
            "temperature": 0.7,
            "thinkingConfig": {"includeThoughts": False}
        }
    }
    
    # Gemini needs longer timeout for complex prompts
    status, data, duration = make_request(url, headers, payload, timeout=120)
    
    if status == 200 and isinstance(data, dict):
        # Parse Gemini's nested response structure
        candidates = data.get('candidates', [])
        if not candidates:
            raise Exception(f"API call failed: No candidates in response. Data: {data}")
        
        candidate = candidates[0]
        finish_reason = candidate.get('finishReason', 'UNKNOWN')
        
        # Handle MAX_TOKENS finish reason
        if finish_reason == 'MAX_TOKENS':
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            if parts:
                text = parts[0].get('text', '')
                if text:
                    print(f"   ⚠️  Warning: Gemini hit MAX_TOKENS limit, returning partial response ({len(text)} chars)")
                    return text
            
            # No usable response, provide helpful error
            usage = data.get('usageMetadata', {})
            total_tokens = usage.get('totalTokenCount', 0)
            thought_tokens = usage.get('thoughtsTokenCount', 0)
            raise Exception(
                f"API call failed: MAX_TOKENS limit reached. "
                f"Total tokens: {total_tokens}, Thinking tokens: {thought_tokens}. "
                f"Try reducing prompt size or increasing max_tokens. Data: {data}"
            )
        
        # Extract text from normal response
        content = candidate.get('content', {})
        if not content:
            raise Exception(f"API call failed: No content in candidate. Finish reason: {finish_reason}. Data: {data}")
        
        parts = content.get('parts', [])
        if not parts:
            raise Exception(
                f"API call failed: No parts in content. Finish reason: {finish_reason}. "
                f"This might indicate the response was truncated. Data: {data}"
            )
        
        text = parts[0].get('text', '')
        if not text and finish_reason != 'STOP':
            raise Exception(f"API call failed: Empty response. Finish reason: {finish_reason}. Data: {data}")
        
        return text
    else:
        raise Exception(f"API call failed: {data}")


# =============================================================================
# FAST MODEL CALLERS (convenience wrappers)
# =============================================================================

def call_claude_fast(prompt, max_tokens=4000, timeout=60):
    """Call Claude Haiku (fast/cheap model). See call_claude for details."""
    return call_claude(prompt, max_tokens, model=CLAUDE_FAST, timeout=timeout)


def call_openai_fast(prompt, max_tokens=4000, timeout=60):
    """Call GPT-4o-mini (fast/cheap model). See call_openai for details."""
    return call_openai(prompt, max_tokens, model=OPENAI_FAST, timeout=timeout)


def call_gemini_fast(prompt, max_tokens=4000):
    """Call Gemini Flash (fast model). Note: Falls back to Pro on this platform."""
    return call_gemini(prompt, max_tokens, model=GEMINI_FAST)


# =============================================================================
# UNIFIED LLM CALLER
# =============================================================================

def call_llm(prompt, max_tokens=4000, model_type=None, fast=False):
    """
    Unified LLM caller - route to appropriate provider based on model_type.
    
    Args:
        prompt: The prompt to send
        max_tokens: Maximum tokens in response
        model_type: Provider/mode selection:
            - "claude"     : Call Claude (capable or fast based on `fast` flag)
            - "openai"     : Call OpenAI (capable or fast based on `fast` flag)
            - "gemini"     : Call Gemini (capable or fast based on `fast` flag)
            - "all"        : Call all 3 capable models, return dict
            - "fast"       : Call all 3 fast models, return dict
            - "all_models" : Call all 6 models, return dict
        fast: If True, use fast/cheap variant (only for single-provider modes)
    
    Returns:
        Response text (single provider) or dict of responses (multi-provider)
    
    Raises:
        Exception: If model_type is unknown or API call fails
    """
    if model_type is None:
        model_type = USE_MODEL.lower()
    
    # Single provider modes
    if model_type == "claude":
        return call_claude_fast(prompt, max_tokens) if fast else call_claude(prompt, max_tokens)
    elif model_type == "openai":
        return call_openai_fast(prompt, max_tokens) if fast else call_openai(prompt, max_tokens)
    elif model_type == "gemini":
        return call_gemini_fast(prompt, max_tokens) if fast else call_gemini(prompt, max_tokens)
    
    # Multi-provider modes
    elif model_type == "all":
        return {
            "claude": call_claude(prompt, max_tokens),
            "openai": call_openai(prompt, max_tokens),
            "gemini": call_gemini(prompt, max_tokens)
        }
    elif model_type == "fast":
        return {
            "claude": call_claude_fast(prompt, max_tokens),
            "openai": call_openai_fast(prompt, max_tokens),
            "gemini": call_gemini_fast(prompt, max_tokens)
        }
    elif model_type == "all_models":
        return {
            "claude_capable": call_claude(prompt, max_tokens),
            "claude_fast": call_claude_fast(prompt, max_tokens),
            "openai_capable": call_openai(prompt, max_tokens),
            "openai_fast": call_openai_fast(prompt, max_tokens),
            "gemini_capable": call_gemini(prompt, max_tokens),
            "gemini_fast": call_gemini_fast(prompt, max_tokens)
        }
    else:
        raise Exception(
            f"Unknown model type: {model_type}. "
            f"Use 'claude', 'openai', 'gemini', 'all', 'fast', or 'all_models'"
        )


# =============================================================================
# SUMMARY & REPORTING
# =============================================================================

def print_model_summary():
    """Print a summary of all available models with measured response times."""
    print("\n" + "=" * 90)
    print("📋 AVAILABLE MODELS ON RAKUTEN AI PLATFORM")
    print("=" * 90)
    
    # Model reference table
    print("""
   ┌─────────────┬────────────────────────────────┬──────────┬─────────────────────────────┐
   │ Provider    │ Model                          │ Type     │ Best For                    │
   ├─────────────┼────────────────────────────────┼──────────┼─────────────────────────────┤
   │ ANTHROPIC   │ claude-opus-4-5-20251101       │ 🧠 Smart │ Complex reasoning, coding   │
   │             │ claude-3-5-haiku-20241022      │ ⚡ Fast  │ Quick tasks, high volume    │
   ├─────────────┼────────────────────────────────┼──────────┼─────────────────────────────┤
   │ OPENAI      │ gpt-5.1                        │ 🧠 Smart │ Complex reasoning, coding   │
   │             │ gpt-4o-mini                    │ ⚡ Fast  │ Quick tasks, cheapest       │
   │             │ o4-mini                        │ ⚡ Fast  │ Reasoning, efficient        │
   ├─────────────┼────────────────────────────────┼──────────┼─────────────────────────────┤
   │ GOOGLE      │ gemini-3-pro-preview           │ 🧠 Smart │ Complex tasks, multimodal   │
   └─────────────┴────────────────────────────────┴──────────┴─────────────────────────────┘
""")
    
    # Speed comparison (only if we have measured times)
    if RESPONSE_TIMES:
        print("   ⚡ SPEED COMPARISON (actual measured response times):")
        print("   ┌─────────────────────────────────┬────────────┬──────────┐")
        print("   │ Model                           │ Response   │ Rank     │")
        print("   ├─────────────────────────────────┼────────────┼──────────┤")
        
        # Sort by response time (fastest first)
        sorted_times = sorted(RESPONSE_TIMES.items(), key=lambda x: x[1])
        
        for rank, (model, duration) in enumerate(sorted_times, 1):
            rank_label = "🥇 FASTEST" if rank == 1 else f"#{rank}"
            model_display = model[:30] if len(model) > 30 else model
            print(f"   │ {model_display:<31} │ {duration:>7.2f}s   │ {rank_label:<8} │")
        
        print("   └─────────────────────────────────┴────────────┴──────────┘")
    else:
        print("   ⚡ SPEED COMPARISON: Run tests first to measure response times")
    
    # Usage examples
    print("""
   Usage in code:
   - call_claude(prompt)       → Opus 4.5 (smart)
   - call_claude_fast(prompt)  → Haiku 3.5 (fast)
   - call_openai(prompt)       → GPT-5.1 (smart)
   - call_openai_fast(prompt)  → GPT-4o-mini (fast)
   - call_gemini(prompt)       → Gemini 3 Pro
   
   - call_llm(prompt, model_type="all")   → All 3 smart models
   - call_llm(prompt, model_type="fast")  → All 3 fast models
""")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("🚀 RAKUTEN AI PLATFORM - API TEST SUITE")
    print("   Testing all available models: Capable 🧠 + Fast ⚡")
    print("=" * 90)
    
    # Run tests for all providers
    test_openai()
    test_anthropic()
    test_gemini()
    
    # Print summary with measured times
    print_model_summary()
    
    print("\n" + "=" * 90)
    print("✨ Testing complete! All models tested.")
    print("=" * 90 + "\n")

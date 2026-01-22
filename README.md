# 🤖 Auto-Heal Pipeline

> **AI-powered code generation that learns from its mistakes**

An iterative self-improvement pipeline that uses LLMs (Claude, GPT, Gemini) to automatically generate and refine data extraction scripts from HTML.

---

## 🎯 What It Does

1. **You provide**: HTML samples + expected values (ground truth)
2. **AI generates**: Python extraction code
3. **Pipeline tests**: Runs code against your samples
4. **AI improves**: Sees failures and fixes them
5. **Repeat**: Until accuracy target is reached

---

## 🚀 Quick Start

### 1. Prepare your data

Put your CSV in `store_data/Store_OCP_Data.csv` with columns:
- `RAW_DOM` - HTML content
- `GROUND_TRUTH_ORDER_ID` - Expected order ID
- `GROUND_TRUTH_SUBTOTAL` - Expected subtotal

### 2. Configure models to test

Edit the `MODELS_TO_TEST` list in `orchestrator.py`:

```python
MODELS_TO_TEST = [
    ("claude_fast", api_test.CLAUDE_FAST, "fast"),     # Fast
    ("openai_fast", api_test.OPENAI_FAST, "fast"),     # Fast
    # ("claude",      api_test.CLAUDE_MODEL, "capable"), # Slower but smarter
    # ("openai",      api_test.OPENAI_MODEL, "capable"), # Slower but smarter
]
```

### 3. Run the pipeline

```bash
python orchestrator.py
```

### 4. Get your script

The best extraction script is saved as `final_{model}_iter{n}.py`

---

## 📁 Project Structure

```
├── orchestrator.py          # 🧠 Main pipeline coordinator
├── api_test.py              # 🔌 LLM API client (Claude, OpenAI, Gemini)
├── script_utils.py          # 🧹 Utilities & cleanup
├── bad_script.py            # 📉 Baseline comparison script
├── prompt_builders/         # 📝 Prompt templates
│   ├── base.py              #    Shared context extraction
│   ├── first_inference.py   #    Initial code generation prompt
│   └── improvement.py       #    Iterative improvement prompt
└── store_data/
    └── Store_OCP_Data.csv   # 📊 Your HTML samples + ground truth
```

---

## 📊 Example Output

```
========================================================================================================================
🚀 AUTO-HEAL PIPELINE v3.0
========================================================================================================================

📂 Loading data from CSV: store_data/Store_OCP_Data.csv
   Found 20 rows in CSV

🎲 Splitting train/test (50/50)...
   Training: 10 rows | Test: 10 rows (holdout)

📋 Ground Truth Presence in HTML:
┌────────────────────┬──────────────────┬──────────────────┐
│ Ground Truth       │ Training Set     │ Test Set         │
├────────────────────┼──────────────────┼──────────────────┤
│ Order ID Present   │  10/10  (100%)   │  10/10  (100%)   │
│ Subtotal Present   │   0/10  (  0%)   │   1/10  ( 10%)   │
└────────────────────┴──────────────────┴──────────────────┘

🤖 PHASE 1: Testing CLAUDE_FAST, OPENAI_FAST
   ✅ OPENAI_FAST: Order ID 100%, Subtotal 0%

🔄 PHASE 2: Iterative Improvement (5 iterations)
   Iteration 1: 80% accuracy
   Iteration 2: 80% accuracy ⭐ BEST (lowest latency)

📊 FINAL RESULTS
   🏆 Winner: OPENAI_FAST iteration 2
   ✅ Best script saved as: final_openai_fast_iter2.py
```

---

## 🔧 Configuration

### Models Available

| Model | Type | Speed | Quality |
|-------|------|-------|---------|
| `claude` | Capable | Slower | Higher |
| `openai` | Capable | Slower | Higher |
| `gemini` | Capable | Slower | Higher |
| `claude_fast` | Fast | Faster | Good |
| `openai_fast` | Fast | Faster | Good |

### Pipeline Settings

In `orchestrator.py`:

```python
MIN_PASS_RATE = 70      # Target accuracy (%)
MAX_ITERATIONS = 5      # Max improvement attempts
```

---

## 🧠 How It Works

### Phase 1: Initial Generation
- Builds prompt with HTML context around ground truth values
- Asks multiple LLMs to generate extraction scripts
- Tests each and picks the best performer

### Phase 2: Iterative Improvement
- Takes the best script
- Shows it the failures with relevant HTML context
- Asks it to fix the issues
- Repeats until accuracy target or max iterations

### Selection Criteria
1. **Primary**: Maximum correct extractions
2. **Secondary**: Lowest latency (if tied)

---

## 📋 Key Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main pipeline - run this |
| `api_test.py` | LLM API calls |
| `script_utils.py` | Utilities |
| `bad_script.py` | Baseline for comparison |
| `prompt_builders/*.py` | Customizable prompts |

---

## 💡 Tips

- **More samples = better results** (20+ recommended)
- **Include edge cases** in your training data
- **Check ground truth presence** - if values aren't in HTML, extraction will fail
- **Test set is sacred** - never tune based on test results

---

## 🔐 Environment Variables (Optional)

| Variable | Description |
|----------|-------------|
| `SSL_CERT_FILE` | Corporate SSL certificate path (auto-detected from `~/rakuten-ca.pem`) |

---

## 📝 License

MIT

---

Built with ❤️ and iterative improvement

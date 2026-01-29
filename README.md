# 🤖 Auto-Heal Pipeline (Multi-Store)

> **AI-powered code generation that learns from its mistakes - now with multi-store support**

An iterative self-improvement pipeline that uses LLMs (Claude, GPT, Gemini) to automatically generate and refine data extraction scripts from HTML. Processes each store independently and generates optimized scripts for each.

---

## 🎯 What It Does

1. **You provide**: CSV with HTML samples + expected values (from multiple stores)
2. **For EACH store**:
   - AI generates Python extraction code
   - Pipeline tests code against that store's samples
   - AI improves by seeing failures
   - Repeat until accuracy target is reached
3. **You get**: One optimized script per store

---

## 🚀 Quick Start

### 1. Prepare your data

Put your CSV in `store_data/Store_OCP_Data.csv` with columns:
- `STORE_ID` - Unique identifier for each store
- `RAW_DOM` - HTML content
- `GROUND_TRUTH_ORDER_ID` - Expected order ID
- `GROUND_TRUTH_SUBTOTAL` - Expected subtotal

**Example CSV structure:**
```csv
STORE_ID,RAW_DOM,GROUND_TRUTH_ORDER_ID,GROUND_TRUTH_SUBTOTAL
target,<html>...,ORD-123,$45.99
target,<html>...,ORD-456,$32.50
walmart,<html>...,WMT-789,$78.00
walmart,<html>...,WMT-012,$15.25
```

### 2. Configure settings

Edit `orchestrator.py`:

```python
# Models to test (comment/uncomment as needed)
MODELS_TO_TEST = [
    ("claude_fast", api_test.CLAUDE_FAST, "fast"),     # Fast
    ("openai_fast", api_test.OPENAI_FAST, "fast"),     # Fast
    # ("claude",      api_test.CLAUDE_MODEL, "capable"), # Slower but smarter
]

# Pipeline settings
MAX_ITERATIONS = 1   # Set to 1 for quick testing, 5 for thorough optimization
MIN_SAMPLES_PER_STORE = 2  # Stores with fewer samples are skipped
```

### 3. Run the pipeline

```bash
python orchestrator.py
```

### 4. Get your scripts

The best extraction script for each store is saved as:
```
final_{store_id}_{model}_iter{n}.py
```

---

## 📁 Project Structure

```
├── orchestrator.py          # 🧠 Main pipeline (multi-store loop)
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
🚀 AUTO-HEAL PIPELINE v4.0 - Multi-Store Support
   Per-store processing | MAX_ITERATIONS=1 | MIN_SAMPLES=2
========================================================================================================================

📂 Loading data from CSV: store_data/Store_OCP_Data.csv
   Found 100 total rows
   Found 5 unique store(s)
   Models to test: CLAUDE_FAST, OPENAI_FAST

========================================================================================================================
🏪 PROCESSING 5 STORE(S)
========================================================================================================================

────────────────────────────────────────────────────────────────────────────────
   [1/5] STORE: target
────────────────────────────────────────────────────────────────────────────────
   📂 Store target: 20 samples
   Training: 10 | Test: 10
   Ground Truth: OID=100%/100%, SUB=10%/0%
   ✅ CLAUDE_FAST ⚡ | OID: 80% | SUB: 0% | Iter: 0 | 97ms

────────────────────────────────────────────────────────────────────────────────
   [2/5] STORE: walmart
────────────────────────────────────────────────────────────────────────────────
   📂 Store walmart: 25 samples
   Training: 12 | Test: 13
   Ground Truth: OID=100%/100%, SUB=85%/80%
   ✅ OPENAI_FAST ⚡ | OID: 90% | SUB: 75% | Iter: 1 | 45ms

... (more stores) ...

========================================================================================================================
📊 FINAL RESULTS - ALL STORES
========================================================================================================================

   📊 Store Results Summary (5 stores processed, 0 skipped):
   Store ID        | Samples  | Model        | OID %   | SUB %   | Both %  | Fields     | Latency    | Iter 
   --------------- | -------- | ------------ | ------- | ------- | ------- | ---------- | ---------- | -----
   walmart         | 25       | openai_fast  |    90%  |    75%  |    70%  | 33/40      |     45.0ms | 1    
   amazon          | 30       | claude_fast  |    85%  |    60%  |    55%  | 29/40      |     67.0ms | 1    
   target          | 20       | claude_fast  |    80%  |     0%  |     0%  |  8/20      |     97.0ms | 0    
   --------------- | -------- | ------------ | ------- | ------- | ------- | ---------- | ---------- | -----
   AVERAGE         |          |              |   85.0% |   45.0% |   41.7% | 70/100     |     69.7ms |

   🏆 Best Model Distribution:
      claude_fast: 3 stores (60%)
      openai_fast: 2 stores (40%)

   📁 Generated Scripts:
      - final_target_claude_fast_iter0.py
      - final_walmart_openai_fast_iter1.py
      - final_amazon_claude_fast_iter1.py
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
MIN_PASS_RATE = 70           # Target accuracy (%) per store
MAX_ITERATIONS = 1           # Max improvement attempts per store
                             # Set to 1 for many stores (lower API costs)
                             # Set to 5 for thorough optimization
MIN_SAMPLES_PER_STORE = 2    # Skip stores with fewer samples
```

---

## 🧠 How It Works

### Per-Store Processing

For EACH unique `STORE_ID` in your CSV:

1. **Filter** - Get only that store's samples
2. **Split** - 50% training / 50% test (holdout)
3. **Generate** - Ask LLMs to create extraction scripts
4. **Test** - Run against training samples
5. **Improve** - Show failures, ask for fixes (up to MAX_ITERATIONS)
6. **Save** - Keep the best script for that store

### Selection Criteria (per store)
1. **Primary**: Maximum correct extractions on TEST set
2. **Secondary**: Lowest latency (if tied)

---

## 📋 Key Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main pipeline with multi-store loop |
| `api_test.py` | LLM API calls |
| `script_utils.py` | Utilities (now includes store_id in naming) |
| `bad_script.py` | Baseline for comparison |
| `prompt_builders/*.py` | Customizable prompts |

---

## 💡 Tips

### For Many Stores (100+)
- Set `MAX_ITERATIONS = 1` to minimize API costs
- Use fast models (`claude_fast`, `openai_fast`)
- Run overnight if needed

### For Few Stores (< 10)
- Set `MAX_ITERATIONS = 5` for thorough optimization
- Can include capable models for higher accuracy

### Data Quality
- **More samples per store = better results** (10+ recommended)
- **Include edge cases** in your training data
- **Check ground truth presence** - if values aren't in HTML, extraction will fail
- Stores with < 2 samples are automatically skipped

---

## 🔐 Environment Variables (Optional)

| Variable | Description |
|----------|-------------|
| `SSL_CERT_FILE` | Corporate SSL certificate path (auto-detected from `~/rakuten-ca.pem`) |


Built with ❤️ and iterative improvement

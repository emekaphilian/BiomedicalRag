# QUICK START: Cohere Integration

**Status**: ✅ **TESTED AND WORKING**
**Test Script**: `stabilization_test_cohere.py`
**Module**: `rag/generator_cohere.py`
**Model**: `command-nightly` (active model, supports biomedical reasoning)

---

## 3-Step Setup

### 1. Get Cohere API Key

Visit: https://dashboard.cohere.com/

Free tier available for testing.

### 2. Set Environment Variable

**PowerShell (Windows)**:
```powershell
$env:COHERE_API_KEY="your-key-here"
```

**Bash/Unix**:
```bash
export COHERE_API_KEY="your-key-here"
```

### 3. Install SDK

```bash
.\venv\Scripts\pip.exe install cohere
```

Or already in requirements.txt:
```bash
.\venv\Scripts\pip.exe install -r requirements.txt
```

---

## Run Test

```bash
$env:COHERE_API_KEY="your-key"
.\venv\Scripts\python.exe stabilization_test_cohere.py
```

---

## What It Does

1. Loads 32,814 biomedical documents from MedQuAD
2. Builds FAISS + BM25 indexes with hybrid fusion
3. Tests 3 biomedical queries with Cohere
4. Generates grounded answers using `command-nightly`
5. Validates evidence citations
6. Reports validation results

---

## Expected Output (All Passing)

```
[Test 1/3] Query: What are the main symptoms of diabetes?
  Retrieved 6 fused results
  Formatted 5 chunks for prompt
  Generated answer (610 chars)
  Validation: PASS
  [OK] GROUNDING VALIDATED

[Test 2/3] Query: What causes Parkinson's disease?
  Retrieved 5 fused results
  Formatted 5 chunks for prompt
  Generated answer (281 chars)
  Validation: PASS
  [OK] GROUNDING VALIDATED

[Test 3/3] Query: What is leukemia?
  Retrieved 6 fused results
  Formatted 5 chunks for prompt
  Generated answer (658 chars)
  Validation: PASS
  [OK] GROUNDING VALIDATED

================================================================================
Tests passed: 3/3
[SUCCESS] All grounding tests passed!
Cohere integration working correctly.
```

---

## Cost

- 3 test queries = ~$0.003 (0.3 cents)
- Run as many times as needed
- Free tier has 20 monthly calls, plenty for testing

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `COHERE_API_KEY not set` | `$env:COHERE_API_KEY="key"` |
| `No module cohere` | `pip install cohere` |
| `Invalid API key` | Check dashboard.cohere.com |
| Model removed error | Use `command-nightly` (active model) |

---

## After Test Passes

1. ✅ **Stabilization Task 1 COMPLETE**: Real LLM integration verified
2. ✅ Proceed with full stabilization phase
3. ✅ Test other queries with different topics
4. ✅ Add comprehensive logging (JSONL format)
5. ✅ Persist indexes to disk (10x speedup)
6. ✅ Create evaluation benchmark (20-50 test queries)

See: `STABILIZATION_PHASE.md`

---

**Ready?** Set your API key and run:
```bash
.\venv\Scripts\python.exe stabilization_test_cohere.py
```

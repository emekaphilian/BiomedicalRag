# IMMEDIATE NEXT STEP: Task 1 - Real LLM Integration

**Priority**: Highest
**Complexity**: Low
**Time**: 30 minutes to 1 hour
**Blocker**: Requires OPENAI_API_KEY

---

## What You're Doing

Converting your pipeline from mock LLM → real LLM (gpt-4o-mini).

**Why**: Validate that real grounding behavior actually works.

---

## Step 1: Set Your API Key

Option A (PowerShell, Windows):
```powershell
$env:OPENAI_API_KEY="sk-..."
```

Option B (Bash/Unix):
```bash
export OPENAI_API_KEY="sk-..."
```

Option C (Python script):
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## Step 2: Create Test Script

Create `stabilization_test_1.py`:

```python
import os
import sys

os.chdir(r"f:\Users\phili\Documents\Projects\LLM-powered-document-Q&A-system-RAG\rag-qa")
sys.path.insert(0, ".")

from rag.generation_pipeline import GroundedGenerationPipeline

# Initialize pipeline with REAL LLM
pipeline = GroundedGenerationPipeline(
    use_mock=False,  # ← Switch to real API
    model="gpt-4o-mini"  # Cheap but capable
)

# Test 3 queries
test_queries = [
    "What are the main symptoms of diabetes?",
    "What causes Parkinson's disease?",
    "What is leukemia?"
]

print("=" * 80)
print("REAL LLM GROUNDING VALIDATION TEST")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}/3] Query: {query}")
    print("-" * 80)
    
    try:
        result = pipeline.generate(query)
        
        print(f"Status: {'PASS' if result['valid'] else 'FAIL'}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        print(f"Citations found: {result.get('citations_found', False)}")
        print(f"\nAnswer (first 300 chars):")
        print(result['answer'][:300] + "...")
        print(f"\nSources: {result.get('sources', [])}")
        
        # Key checks
        if result['valid']:
            print("\n✓ VALIDATION PASSED")
        else:
            print("\n✗ VALIDATION FAILED")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete. Check output above.")
print("=" * 80)
```

---

## Step 3: Run Test

```powershell
$env:OPENAI_API_KEY="sk-..."
.\venv\Scripts\python.exe stabilization_test_1.py
```

---

## What to Look For

✅ **Good Signs**:
- No API errors
- Answer includes medical information
- Citations are present (Source: CDC, etc.)
- Answer quality matches retrieved evidence
- Validation passes

❌ **Bad Signs**:
- API rate limit errors (quota exceeded)
- Hallucinated medical claims
- Missing citations
- Validation failures
- Timeout (> 10 seconds)

---

## Expected Output

```
================================================================================
REAL LLM GROUNDING VALIDATION TEST
================================================================================

[1/3] Query: What are the main symptoms of diabetes?
--------------------------------------------------------------------------------
Status: PASS
Reason: Passed
Citations found: True

Answer (first 300 chars):
Based on the retrieved biomedical evidence from MedQuAD:

Diabetes is characterized by several key symptoms that are primarily related to 
elevated blood glucose levels. The main symptoms include:

1. **Polyuria** (increased urination)
2. **Polydipsia** (increased thirst)
3. **Fatigue and weakness**
...

Sources: ['CDC', 'MedlinePlus']

✓ VALIDATION PASSED

[2/3] Query: What causes Parkinson's disease?
...
```

---

## If It Fails

### Error: "No API key provided"
```
Solution: Make sure OPENAI_API_KEY is set BEFORE running Python
$env:OPENAI_API_KEY="sk-..."
```

### Error: "Invalid API key"
```
Solution: Check that key is correct
Visit: https://platform.openai.com/account/api-keys
```

### Error: "Rate limit exceeded"
```
Solution: Wait a few minutes, try again
You may have hit quota limit
```

### Error: "Timeout"
```
Solution: API is slow or query is complex
Try again with different query
```

### LLM Refuses to Answer
```
Solution: This might be safety filter or insufficient evidence
Try different query
```

---

## Next: After This Succeeds

If Task 1 passes:

1. ✅ Verify grounding works (Task 2)
2. ✅ Test failure cases (Task 3)
3. ✅ Add logging (Task 4)
4. ✅ Persist indexes (Task 5)
5. ✅ Create benchmark queries (Task 6)

See `STABILIZATION_PHASE.md` for full roadmap.

---

## Cost Estimate

**gpt-4o-mini costs**:
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

For this test (3 queries):
- ~1500 tokens total
- **Cost: ~$0.001 (less than 1 cent)**

Safe to run multiple times.

---

## Time Investment

This task:
- Setup: 5 min
- Run test: 2 min
- Analysis: 5 min
- **Total: 12 minutes**

Worth it for validation.

---

**Ready? Run this:**

```powershell
$env:OPENAI_API_KEY="sk-..."
.\venv\Scripts\python.exe stabilization_test_1.py
```

Then report results.

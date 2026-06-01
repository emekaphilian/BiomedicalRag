# Retrieval Error Analysis - Complete Workflow

## Phase 1: Run Diagnostic

```powershell
$env:COHERE_API_KEY="your-cohere-api-key-here"
.\venv\Scripts\python.exe stabilization_task3c_error_analysis.py
```

**Runtime:** ~15 minutes

**Output files:**
- `retrieval_debug.jsonl` — Full chunk texts for manual inspection
- `dense_results.json` — Dense retrieval metrics
- `bm25_results.json` — BM25 sparse retrieval metrics
- `rrf_results.json` — RRF fusion metrics
- `reranked_results.json` — RRF + cross-encoder metrics
- `retrieval_comparison.json` — Summary comparison by method and category

---

## Phase 2: Create Analysis Template

```powershell
.\venv\Scripts\python.exe create_analysis_template.py
```

**Output:** `retrieval_analysis.csv`

This creates a blank CSV with all 18 queries. You'll fill it in based on `retrieval_debug.jsonl`.

---

## Phase 3: Manual Analysis (You Do This)

### For Each of 18 Queries:

1. **Open** `retrieval_debug.jsonl` in a text editor
2. **Find** the query record
3. **Look at** `retrieved_chunks[0:5]` (top 5 ranked results)
4. **Read** each chunk's `text` field
5. **Check** the `has_relevant_keyword` flag
6. **Ask yourself:**
   - Is the relevant information present in top 5? → YES/NO
   - If NO, why? (embedding mismatch, keyword mismatch, chunk quality, etc.)
7. **Fill CSV:**
   - `Was_Relevant_Info_Present`: YES or NO
   - `Error_Type`: Pick from list
   - `Details`: What specifically went wrong
   - `Suggested_Fix`: How to fix it

### Error Type Options

| Error Type | Meaning | Example |
|------------|---------|---------|
| **Embedding_mismatch** | Query and doc use different vocabulary | Query: "heart attack" → Docs mention only "myocardial infarction" |
| **Keyword_mismatch** | Key term not in BM25 index | Query: "Type 2 diabetes" → Docs: "T2DM" or "T2D" |
| **Chunk_too_small** | Relevant info in corpus but split across chunks | Info spread: chunk 1=symptoms, chunk 5=causes, neither complete |
| **Chunk_too_large** | Too much noise in context | Chunk mixes unrelated diseases |
| **RRF_ranking** | Relevant chunk present but ranked low | Dense and BM25 disagree on rank |
| **Reranker_issue** | Cross-encoder ranked relevant doc DOWN | Was rank 2 before, rank 8 after |
| **Dataset_gap** | Answer not in corpus at all | Query asks about drug not in dataset |
| **Ambiguous_query** | Query has multiple valid interpretations | "Why am I tired?" → too many possible causes |
| **None** | No error; successfully retrieved | Found and ranked correctly |

---

## Phase 4: Interpret Results

### Read `retrieval_comparison.json`

Example output:
```json
{
  "by_method": {
    "dense": 0.522,
    "bm25": 0.489,
    "rrf": 0.555,
    "reranked": 0.532
  },
  "by_category": {
    "A_FACTOID": {
      "dense": 0.800,
      "bm25": 0.600,
      "rrf": 0.850,
      "reranked": 0.800
    },
    "E_ADVERSARIAL": {
      "dense": 0.100,
      "bm25": 0.050,
      "rrf": 0.150,
      "reranked": 0.100
    }
  }
}
```

**What to look for:**

1. **Which method is worst?**
   - If Dense < BM25 → Dense embeddings are weak (candidate: replace with BioLORD)
   - If BM25 < Dense → Keyword matching is weak (candidate: add synonym expansion)
   - If both < 0.5 → Both are bottlenecks

2. **Reranker helped or hurt?**
   - If reranked > rrf → Keep the reranker
   - If reranked < rrf → Reranker is harmful, remove it

3. **Which categories fail?**
   - A_FACTOID high → Embedding good for simple facts
   - E_ADVERSARIAL low → Embedding struggles with nuanced/negative queries
   - C_MULTIHOP low → Need multi-document retrieval

---

## Phase 5: Choose Fix

Based on your analysis, pick ONE primary bottleneck:

### Fix 1: Replace Embeddings (if Dense is bottleneck)

Current:
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Better for biomedical:
```python
# Option A: Lightweight biomedical
model = SentenceTransformer("allenai/specter")

# Option B: Medium (recommended)
model = SentenceTransformer("pritamdeka/BioBERT-mnli")

# Option C: Heavy biomedical
model = SentenceTransformer("biogpt")

# Option D: Your proposal recommends
model = SentenceTransformer("BioLORD-2023")
```

**Impact:** Can improve Recall@5 by 0.10-0.20

---

### Fix 2: Add Medical Synonyms (if BM25 is bottleneck)

Current BM25 query:
```
What causes asthma?
```

Better:
```
What causes asthma? OR reactive airway disease? OR RAD?
```

Create medical synonym dictionary:

```python
MEDICAL_SYNONYMS = {
    "heart attack": ["myocardial infarction", "MI", "acute MI"],
    "high blood pressure": ["hypertension", "HTN"],
    "asthma": ["reactive airway disease", "RAD"],
    "type 2 diabetes": ["T2DM", "T2D", "diabetes mellitus type 2"],
    # ... add more
}
```

Then expand query before BM25 search.

**Impact:** Can improve Recall@5 by 0.05-0.15

---

### Fix 3: Tune RRF Parameter k (if Ranking is bottleneck)

Current (in `rag/fusion.py`):
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
```

Test different k values:
```python
for k in [10, 30, 60, 100]:
    # Run evaluation
    # Measure Recall@5
    # Pick best k
```

Each k trades:
- k=10: More weight on top ranks
- k=60: Balanced
- k=100: More weight on full ranking

**Impact:** Can improve Recall@5 by 0.02-0.05

---

### Fix 4: Remove Reranker (if reranker hurts)

If reranker is degrading recall, try removing it:

In your evaluation, just use RRF fusion directly without cross-encoder.

**Impact:** Restore lost 0.02-0.03 points

---

## Phase 6: Implement Best Fix

Once you choose, implement and re-run evaluation:

```powershell
# Edit rag/vectorstore.py (if replacing embeddings)
# OR edit rag/fusion.py (if tuning RRF)
# OR edit BM25 query expansion (if adding synonyms)

# Re-run evaluation
.\venv\Scripts\python.exe stabilization_task3c_error_analysis.py

# Check new retrieval_comparison.json
# Did Recall@5 improve?
```

**Success target:** Recall@5: 0.555 → 0.65+ (target: 0.75)

---

## Phase 7: Document Findings

Update this section with your actual results:

### Analysis Results

**Date:** [When you completed analysis]

**Method Performance (Recall@5):**
- Dense: ___
- BM25: ___
- RRF: ___
- Reranked: ___

**Bottleneck Identified:** [Which method is worst?]

**Worst Performing Categories:**
1. ___ (Recall@5: ___)
2. ___ (Recall@5: ___)
3. ___ (Recall@5: ___)

**Top Failure Cases:**
1. Query: ___ → Error: ___ → Fix: ___
2. Query: ___ → Error: ___ → Fix: ___
3. Query: ___ → Error: ___ → Fix: ___

**Recommended Improvement:** [Which of 4 fixes above?]

**Expected Impact:** Recall@5: 0.555 → ___

---

## Summary

This workflow takes you from:
- ❌ "The system scored 100% so it must be good" (false metric)
- ✅ "I know exactly why Recall@5 is 0.555, and here's how to fix it" (diagnostic clarity)

The 4 experiments show you which component is broken. The manual analysis shows you why. The fixes are targeted. The re-evaluation proves they work.

This is how research systems are built — measure, diagnose, fix, measure again.

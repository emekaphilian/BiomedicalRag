# Biomedical Embeddings Upgrade Guide

**Goal:** Switch from generic SentenceTransformers to BioBERT embeddings to improve Recall@5 from 59.7% to 65-68%

---

## STEP 1: Install BioBERT-Based Model

Replace generic embeddings with **Specter** (optimized for scientific papers) or **BioBERT**:

```bash
# Add to requirements.txt:
sentence-transformers>=2.2.2
# (already installed, just use different model)

# Or if needed:
pip install sentence-transformers --upgrade
```

**Model Options (ranked by biomedical fitness):**

| Model | Size | Domain | Recall@5 Est. | Latency |
|-------|------|--------|---------------|---------|
| `specter` | 768d | Scientific papers | +6-8% | 50ms |
| `allenai/specter-v2` | 768d | General academic | +5-7% | 50ms |
| `dmis-lab/biobert-base-cased` | 768d | Biomedical (PubMed) | +5-7% | 50ms |
| `sentence-t5-large` | 768d | General | +3-5% | 100ms |
| `all-MiniLM-L6-v2` (current) | 384d | General | Baseline | 20ms |

**Recommendation:** Use `allenai/specter` — best balance of biomedical grounding and size.

---

## STEP 2: Modify rag/embeddings.py

**Current code:**
```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # MiniLM dimension
```

**Updated code:**
```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="allenai/specter"):
        self.model = SentenceTransformer(model_name)
        # Specter uses 768d embeddings
        self.embedding_dim = 768
    
    def encode(self, texts, **kwargs):
        """Encode texts to embeddings"""
        return self.model.encode(texts, **kwargs)
```

**Rationale:** Specter is trained on 11M+ scientific papers, understands medical terminology natively.

---

## STEP 3: Re-Index FAISS

```bash
# Delete old index (IMPORTANT!)
rm -r vectorstore_index/

# Re-run indexing script:
python rag/ingestion.py --model allenai/specter

# Or manually in Python:
from rag.embeddings import EmbeddingModel
from rag.vectorstore import FAISSVectorStore

embedder = EmbeddingModel(model_name="allenai/specter")
vectorstore = FAISSVectorStore(
    embeddings=embedder,
    doc_path="data/processed/chunks.jsonl"
)
vectorstore.save("vectorstore_index/")
```

**Timing:** ~10-15 minutes for re-indexing (depends on corpus size)

---

## STEP 4: Update Retrieval Config

Edit `rag/hybrid_retriever.py`:

```python
class HybridRetriever:
    def __init__(self, vectorstore_path="vectorstore_index/"):
        # Embedder now uses Specter by default
        self.embedder = EmbeddingModel(model_name="allenai/specter")
        self.vectorstore = FAISSVectorStore.load(vectorstore_path)
        # Everything else stays the same
```

---

## STEP 5: Run Evaluation

```bash
# Re-run retrieval evaluation with new embeddings:
python evaluate_retrieval.py \
    --test_set data/test_set.jsonl \
    --output_dir results/biobert_upgrade/

# Compare results:
python -c "
import json
old = json.load(open('reranked_results.json'))
new = json.load(open('results/biobert_upgrade/reranked_results.json'))

old_r5 = sum([m['metrics']['recall@5'] for m in old]) / len(old)
new_r5 = sum([m['metrics']['recall@5'] for m in new]) / len(new)

print(f'Old Recall@5: {old_r5:.4f}')
print(f'New Recall@5: {new_r5:.4f}')
print(f'Improvement: +{(new_r5 - old_r5)*100:.1f}%')
"
```

---

## STEP 6: UMLS Synonym Expansion (Optional, +3-5%)

If BioBERT upgrade alone doesn't reach 75%, add medical synonym expansion:

```python
# rag/query_expansion.py (NEW FILE)

def expand_query_with_umls(query):
    """Expand query with medical synonyms from UMLS"""
    umls_db = load_umls_database()
    
    expanded_terms = []
    for term in query.split():
        # Find synonyms (e.g., "Parkinson's" → "Parkinsonism", "Lewy body disease")
        synonyms = umls_db.get_synonyms(term)
        expanded_terms.extend([term] + synonyms[:2])  # Add top 2 synonyms
    
    return " ".join(expanded_terms)

# Usage in BM25Retriever:
class BM25Retriever:
    def retrieve(self, query, k=10):
        expanded_query = expand_query_with_umls(query)  # NEW
        return self.bm25.get_top_k(expanded_query, k)
```

**UMLS Data Sources:**
- **Option A (Free):** Use UMLS-lite (100K concepts) — faster, lighter
- **Option B (Better):** Full UMLS (4M+ concepts) — requires registration at NIH

---

## STEP 7: Expected Impact

### Before (Current)
```
Method     | Recall@5 | Precision@5 | MRR
-----------|----------|-------------|------
Dense      | 0.597    | 0.789       | 0.796
BM25       | 0.556    | 0.478       | 0.577
RRF        | 0.637    | 0.644       | 0.796
Reranked   | 0.689    | 0.722       | 0.843
TARGET     | 0.750    | 0.600       | 0.650
```

### After (With BioBERT + Reranker)
```
Method     | Recall@5 | Precision@5 | MRR
-----------|----------|-------------|------
Dense      | 0.640+   | 0.810       | 0.820
Reranked   | 0.720+   | 0.750       | 0.850
TARGET     | 0.750    | 0.600       | 0.650
STATUS     | ⚠️ CLOSE | ✅ PASS     | ✅ PASS
```

**To reach 75% target:** BioBERT (68.9% → 72-73%) + larger k-pool (→ 74-75%) ✅

---

## Troubleshooting

### Q: FAISS index incompatible after embedding change?
**A:** Delete the index and rebuild. Different embedding dimensions (384 → 768) are incompatible.

### Q: Model download fails?
**A:** Specter is auto-downloaded by HuggingFace. Requires ~500MB disk space.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("allenai/specter")  # Auto-downloads to ~/.cache/huggingface/
```

### Q: Latency increased from 20ms to 50ms?
**A:** Expected. Larger embeddings (768d vs 384d) take ~2.5x longer, but retrieval accuracy improves significantly.

### Q: Recall improved but Precision@5 dropped?
**A:** Reranker stage will fix this. Precision@5 typically improves after reranking.

---

## Timeline

| Phase | Task | Time | Expected Result |
|-------|------|------|-----------------|
| **1** | Install Specter, re-index | 1 hour | Recall@5: 59.7% → 65-68% |
| **2** | Run evaluation | 15 min | Benchmark new baseline |
| **3** | Add UMLS expansion (if needed) | 2 hours | Recall@5: 65-68% → 69-72% |
| **4** | Increase k to 50 (if needed) | 30 min | Recall@5: 69-72% → 71-75% ✅ |

**Total effort:** 2-4 hours to reach 75% target.

---

## Files to Modify

1. `rag/embeddings.py` — Change model name and embedding_dim
2. `requirements.txt` — Ensure sentence-transformers is latest
3. `rag/ingestion.py` — Re-index with new model
4. `rag/hybrid_retriever.py` — Update model config

**Validation:**
- ✅ Old results: `reranked_results.json` (68.9% Recall@5)
- ✅ New results: `results/biobert_upgrade/reranked_results.json` (target: 72%+)

---

## Next Steps

1. Execute Steps 1-4 (embedding upgrade)
2. Run evaluation
3. Report improvement
4. If Recall@5 < 75%, proceed to UMLS expansion


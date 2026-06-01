# RAG Medical Chatbot — Evaluation Framework
## University of Aveiro | ACM Student Project Conference 2026

---

## Files

| File | Purpose |
|---|---|
| `build_test_set.py` | **Run once.** Splits MedQuAD, adds edge cases, SHA-256 locks the test set |
| `verify_test_set.py` | **Run before every eval.** Confirms test set is unmodified |
| `evaluate_retrieval.py` | Tier 1 — Recall@5/10, Precision@5, MRR, latency |
| `evaluate_generation.py` | Tier 2 — EM, F1, Grounding Rate, Hallucination Rate, Refusal Accuracy |
| `run_evaluation.py` | **Master runner** — calls both tiers, generates HTML report |

---

## Step 1 — Build & lock the test set (run ONCE)

```bash
pip install -r requirements_eval.txt

python build_test_set.py \
    --medquad_dir  /path/to/MedQuAD \
    --output_dir   data/ \
    --seed         42
```

**What it does:**
- Parses all MedQuAD XML files
- Removes duplicates
- Stratified split → `indexed_corpus.jsonl` (your FAISS/BM25 corpus), `dev_set.jsonl` (tuning), `test_set.jsonl` (evaluation)
- Appends 50 edge-case questions (15 pre-written + 35 placeholders)
- Writes `test_set.lock` with SHA-256 fingerprint

**After this step:**
1. Open `data/edge_cases.jsonl`
2. Fill in all 35 placeholder questions (look for `[PLACEHOLDER]`)
3. Re-run `build_test_set.py` with `--skip_medquad` to regenerate the lock on the final edge cases
4. **Never modify `test_set.jsonl` again**

---

## Step 2 — Annotate supporting_docs (important for retrieval metrics)

For retrieval evaluation to work, each test question needs `supporting_docs` filled:

```json
{
  "id": "medquad_123",
  "question": "What are the symptoms of type 1 diabetes?",
  "answer": "...",
  "supporting_docs": ["QA_Medical_Specialty_MedlinePlus_diabetes.xml", "PMID_12345678"]
}
```

For MedQuAD questions, the `file` field already tells you the source XML. Copy it into `supporting_docs`.

---

## Step 3 — Run evaluation

```bash
# Full evaluation (both tiers):
python run_evaluation.py \
    --test_set      data/test_set.jsonl \
    --retriever_url http://localhost:8000 \
    --rag_url       http://localhost:8000 \
    --cohere_key    $COHERE_API_KEY \
    --output_dir    results/

# Retrieval only (faster, no API cost):
python run_evaluation.py --tier retrieval \
    --test_set data/test_set.jsonl \
    --retriever_url http://localhost:8000

# Quick dev test (first 20 questions):
python run_evaluation.py --limit 20 --cohere_key $COHERE_API_KEY
```

The test set integrity check runs automatically at the start. If the file was
modified, the evaluation aborts.

---

## Retriever API contract

Your retriever must expose:

```
POST /retrieve
Body:  { "query": "...", "config": "dense"|"sparse"|"hybrid", "k": 10 }
Response: { "results": [{"doc_id": "...", "score": 0.9, "text": "...", "source": "..."}, ...] }
```

## RAG system API contract

```
POST /query
Body:  { "query": "...", "variant": "baseline_a"|"baseline_b"|"baseline_c" }
Response: { "answer": "...", "sources": [...], "grounded": true, "context": "...", "latency_ms": 200 }
```

---

## Proposal targets

| Metric | Target | Fail threshold |
|---|---|---|
| Recall@5 | ≥ 0.75 | < 0.60 |
| Recall@10 | ≥ 0.85 | < 0.70 |
| Precision@5 | ≥ 0.60 | < 0.45 |
| MRR | ≥ 0.65 | < 0.50 |
| Latency p95 | < 300ms | > 800ms |
| Exact Match | ≥ 35% | < 20% |
| F1 Score | ≥ 0.55 | < 0.35 |
| Grounding Rate | ≥ 80% | < 60% |
| Hallucination Rate | < 10% | > 25% |
| Refusal Accuracy | ≥ 90% | < 70% |

---

## Output files

After a run, `results/` will contain:
- `retrieval_eval_<timestamp>.json` — Tier 1 metrics
- `generation_eval_<timestamp>.json` — Tier 2 summary
- `generation_eval_log_<timestamp>.jsonl` — per-question answers (for spot-check)
- `eval_report_<timestamp>.html` — combined HTML report (open in browser)

---

## Important rules

1. **Never tune on `test_set.jsonl`** — use `dev_set.jsonl` for all hyperparameter decisions
2. **Run `verify_test_set.py` before every evaluation** — or use `run_evaluation.py` which does it automatically
3. **Manual spot-check** — after each run, open `generation_eval_log_*.jsonl` and manually verify the ~10% of questions flagged as `spot_check`
4. **Lock the config before Phase 4** — once you pick the best hyperparameters on the dev set, freeze them and do one final test set run

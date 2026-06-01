# BLOCK 7: EVALUATION FRAMEWORK (NEXT PHASE)

**Status**: Recommended next immediate goal after Block 6B is verified.

---

## Why Evaluation Framework Matters Now

At this stage:
- ✓ Retrieval is working (hybrid dense + sparse + reranking)
- ✓ Generation is working (LLM + structured prompts)
- ✓ Safety is integrated (pre- and post-generation)

**BUT**: You don't know how *well* they work. Without measurement:

```text
You cannot:
- Compare performance between models
- Detect regressions
- Optimize hyperparameters
- Verify medical safety compliance
- Benchmark against baselines
```

---

## What Block 7 Will Build

### Core Metrics

#### 1. **Retrieval Metrics**
```
Recall@K        - What % of relevant docs appear in top-K?
Precision@K     - What % of top-K results are relevant?
MRR (Mean Reciprocal Rank) - How soon does first relevant result appear?
NDCG (Normalized Discounted Cumulative Gain) - Quality of ranking
```

**Why matter**: Tells you if retriever is finding the right documents.

#### 2. **Generation Metrics (Biomedical-Specific)**
```
Faithfulness    - Does answer match retrieved evidence? (no hallucinations)
Citation Accuracy - Do cited sources actually support claims?
Hallucination Rate - % of answers with unsupported claims
Grounding Score - % of medical claims with citations
```

**Why matter**: Medical domain cannot tolerate fabrication.

#### 3. **End-to-End System Metrics**
```
Answer Correctness - Does the system answer the query correctly?
Safety Compliance - Does it refuse unsafe queries?
Latency - How fast is the pipeline?
Token Efficiency - Average tokens used per answer
```

---

## Evaluation Dataset Structure

### Recommended Format

```yaml
query: "What are the main symptoms of diabetes?"
query_type: "symptom"
expected_sources:
  - source: "CDC"
    snippet: "polyuria, polydipsia, fatigue"
  - source: "MedlinePlus"
    snippet: "high blood glucose"
correct_answer: "Increased thirst, frequent urination, fatigue, blurred vision"
unsafe: false
difficulty: "medium"
```

### Where to Get Test Data

1. **Build from MedQuAD itself**
   - Extract all question-answer pairs as ground truth
   - Use Q&A pairs as expected answers

2. **Manual annotation** (best for medical)
   - Pick 50-100 diverse queries
   - Have medical expert validate answers
   - Annotate relevant source segments

3. **Existing biomedical QA benchmarks**
   - BioASQ (2.2K biomedical questions)
   - MedQuAD test set (if available)
   - TREC-CTS (medical QA)

---

## Implementation Plan

### Phase 7.1: Retrieval Evaluation
```python
from evaluation.retrieval_metrics import RetrievalEvaluator

evaluator = RetrievalEvaluator()

for query in test_queries:
    retrieved = pipeline.retrieve(query, top_k=10)
    relevant_docs = ground_truth[query]["relevant_docs"]
    
    recall = evaluator.recall_at_k(retrieved, relevant_docs, k=10)
    precision = evaluator.precision_at_k(retrieved, relevant_docs, k=10)
    mrr = evaluator.mean_reciprocal_rank(retrieved, relevant_docs)
    
    print(f"Query: {query}")
    print(f"  Recall@10: {recall:.3f}")
    print(f"  Precision@10: {precision:.3f}")
    print(f"  MRR: {mrr:.3f}")
```

### Phase 7.2: Generation Evaluation
```python
from evaluation.generation_metrics import GenerationEvaluator

evaluator = GenerationEvaluator()

for query in test_queries:
    result = pipeline.generate(query)
    expected = ground_truth[query]["answer"]
    
    faithfulness = evaluator.faithfulness(
        answer=result["answer"],
        context=result["context"],
        model="gpt-4o"  # Use LLM for semantic similarity
    )
    
    hallucination_rate = evaluator.hallucination_rate(
        answer=result["answer"],
        context=result["context"]
    )
    
    citation_accuracy = evaluator.citation_accuracy(
        answer=result["answer"],
        sources=result["sources"],
        context=result["context"]
    )
    
    print(f"Query: {query}")
    print(f"  Faithfulness: {faithfulness:.3f}")
    print(f"  Hallucination Rate: {hallucination_rate:.3f}")
    print(f"  Citation Accuracy: {citation_accuracy:.3f}")
```

### Phase 7.3: Baseline Comparison
```python
from evaluation.baselines import BaselineRetriever, BaselineGenerator

# Baseline 1: Dense retrieval only (no reranking)
baseline_dense = BaselineRetriever(use_reranking=False)
# Baseline 2: Simple lexical BM25 (no fusion)
baseline_bm25 = BaselineRetriever(use_bm25_only=True)
# Baseline 3: Generic LLM (no grounding prompt)
baseline_generic_llm = BaselineGenerator(use_grounding_prompt=False)

# Compare your system vs baselines
results = {
    "Your System": evaluate_pipeline(pipeline),
    "Dense Only": evaluate_pipeline(baseline_dense),
    "BM25 Only": evaluate_pipeline(baseline_bm25),
    "Generic LLM": evaluate_pipeline(baseline_generic_llm),
}

# Print comparison table
print_comparison_table(results)
```

---

## Files to Create in Block 7

```
evaluation/
  ├── retrieval_metrics.py    [NEW] Recall@K, Precision@K, MRR, NDCG
  ├── generation_metrics.py   [NEW] Faithfulness, Hallucination, Citation Accuracy
  ├── baselines.py            [NEW] Baseline implementations
  └── evaluator.py            [NEW] Main evaluation orchestrator

tests/
  └── test_evaluation.py      [NEW] Full evaluation harness
```

---

## Recommended Evaluation Dataset Size

| Phase | Dataset Size | Purpose |
|-------|--------------|---------|
| Development | 20 queries | Quick iteration |
| Validation | 100 queries | Tuning hyperparameters |
| Testing | 500+ queries | Final metrics, baseline comparison |
| Production | Continuous | Monitor system in production |

---

## Success Criteria for Block 7

### Retrieval Performance
- Recall@10 ≥ 0.80 (find 80% of relevant docs)
- Precision@10 ≥ 0.70 (70% of top-10 are relevant)
- MRR ≥ 0.75 (relevant doc typically in top 3-4)

### Generation Performance
- Faithfulness ≥ 0.85 (85% of answers match evidence)
- Hallucination Rate ≤ 0.10 (≤10% answers with unsupported claims)
- Citation Accuracy ≥ 0.90 (90% of citations are correct)

### System Performance
- Latency ≤ 2.0 seconds (for real-time deployment)
- Token Efficiency ≤ 1000 tokens per answer (cost control)
- Safety Compliance: 100% refusal of unsafe queries

---

## Key Implementation Details

### Faithfulness Scoring
```python
def faithfulness(answer, context):
    """
    Does the answer only use information from context?
    
    Method 1 (Simple): Check for keywords from context in answer
    Method 2 (Semantic): Use LLM to judge semantic alignment
    Method 3 (Factual): Use claim extraction + semantic matching
    """
    # Recommended for medical: Method 2 (LLM-based)
    prompt = f"""
    Context: {context}
    Answer: {answer}
    
    Is this answer faithful to the context? 
    (1=Not faithful, 5=Completely faithful)
    """
    score = llm_judge(prompt)
    return score / 5.0
```

### Hallucination Detection
```python
def hallucination_rate(answer, context):
    """
    What % of answer contains information NOT in context?
    
    Steps:
    1. Extract key claims from answer
    2. Check each claim against context
    3. Count unsupported claims
    4. Compute rate = unsupported / total
    """
    claims = extract_claims(answer)
    supported = [
        c for c in claims 
        if is_supported_by_context(c, context)
    ]
    return 1.0 - (len(supported) / len(claims))
```

### Citation Accuracy
```python
def citation_accuracy(answer, sources, context):
    """
    Do the cited sources actually appear in context?
    Do they support the claims made?
    
    For medical: 
    - Reject claims without citations
    - Verify source is in context
    - Check semantic alignment between claim and source
    """
    cited_sources = extract_sources(answer, sources)
    correct = 0
    for claim, source in zip(extract_claims(answer), cited_sources):
        if source in context and supports(source, claim):
            correct += 1
    return correct / len(cited_sources) if cited_sources else 0.0
```

---

## Benchmark Comparison Example

After Block 7 completes, you should have results like:

```
Retrieval Performance:
                        Recall@10  Precision@10   MRR
Your System (Hybrid)    0.92       0.85          0.88
Baseline (Dense Only)   0.75       0.68          0.70
Baseline (BM25 Only)    0.68       0.72          0.65

Generation Performance:
                        Faithfulness  Hallucination  Citation Acc
Your System (Grounded)  0.91          0.08           0.94
Baseline (Generic LLM)  0.62          0.35           0.45

System Performance:
                        Latency  Token Count  Safety Compliance
Your System             1.2s     850 tokens   100%
Baseline (Generic LLM)  0.8s     2100 tokens  0%
```

---

## Timeline & Next Steps

### Recommended Sequence
1. **Week 1**: Build retrieval evaluation metrics (recall, precision, MRR)
2. **Week 2**: Build generation evaluation metrics (faithfulness, hallucination, citation)
3. **Week 3**: Create baseline implementations
4. **Week 4**: Build evaluation harness and run comprehensive benchmarks

### After Block 7
- **Block 8**: Production deployment (API server, monitoring, compliance logging)
- **Block 9**: Safety & compliance (HIPAA readiness, liability assessment)
- **Block 10**: Advanced features (query expansion, synonym normalization, UMLS integration)

---

## Key Files to Reference

When building Block 7, reference:
- `evaluation/retrieval_metrics.py` - Already started
- `evaluation/generation_metrics.py` - Already started
- `evaluation/baselines.py` - Already started
- Medical evaluation papers (BioASQ, TREC-CTS)

---

## Critical Decision Point

### Question for You:
After Block 7 evaluation is complete and shows metrics:

**Will you**:
1. **Optimize hyperparameters** based on evaluation results?
2. **Move to production** if metrics meet thresholds?
3. **Add advanced features** (UMLS normalization, synonym expansion, etc.)?
4. **Establish compliance** (HIPAA audit, medical liability)?

This decision should be based on:
- Your evaluation metrics results
- Your use case (research vs. clinical deployment)
- Your timeline and resources
- Your risk tolerance for medical information systems

**Block 7 will give you the data to make this decision.**

---

## Ready for Block 7?

Run this first to confirm current system is stable:

```bash
python tests/test_generation.py  # Should pass
python tests/test_reranking.py   # Should pass
python tests/test_hybrid_retrieval.py  # Should pass
```

Then proceed to implement Block 7 evaluation framework.

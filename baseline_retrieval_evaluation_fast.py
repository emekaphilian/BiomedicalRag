"""
baseline_retrieval_evaluation_fast.py
=====================================
Fast baseline retrieval evaluation.

Uses pre-computed dense/sparse retrieval results instead of re-ranking from scratch.
This evaluates the LOCKED TEST SET against baseline metrics.

Expected run time: <5 minutes
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.retrieval_metrics import RetrievalMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_test_set(filepath):
    """Load locked test set."""
    questions = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def main():
    print("=" * 70)
    print("BASELINE RETRIEVAL EVALUATION (all-MiniLM-L6-v2)")
    print("=" * 70)
    
    # Load locked test set
    logger.info("Loading locked test set...")
    test_questions = load_test_set("data/test_set.jsonl")
    logger.info(f"Loaded {len(test_questions)} test questions")
    
    # Check for pre-computed results
    dense_results_file = "dense_results.json"
    
    if not Path(dense_results_file).exists():
        logger.error(f"Baseline results not found: {dense_results_file}")
        logger.error("Run: python stabilization_task3_retrieval_evaluation.py")
        return
    
    # Load pre-computed dense results
    logger.info(f"Loading pre-computed retrieval results from {dense_results_file}...")
    with open(dense_results_file) as f:
        dense_results = json.load(f)
    
    logger.info(f"Loaded {len(dense_results)} pre-computed results\n")
    
    # Map results by query
    results_map = {item["query"]: item for item in dense_results}
    
    # Evaluate on locked test set
    print("=" * 70)
    print("EVALUATING AGAINST LOCKED TEST SET")
    print("=" * 70 + "\n")
    
    results = []
    recall_at_5 = []
    recall_at_10 = []
    precision_at_5 = []
    mrr_scores = []
    ndcg_at_5 = []
    
    for i, question in enumerate(test_questions, 1):
        q_text = question["question"]
        
        # Get pre-computed results
        if q_text in results_map:
            metrics = results_map[q_text]["metrics"]
            
            r5 = metrics.get("recall@5", 0.0)
            r10 = metrics.get("recall@10", 0.0)
            p5 = metrics.get("precision@5", 0.0)
            mrr = metrics.get("mrr", 0.0)
            ndcg5 = metrics.get("ndcg@5", 0.0)
            
            recall_at_5.append(r5)
            recall_at_10.append(r10)
            precision_at_5.append(p5)
            mrr_scores.append(mrr)
            ndcg_at_5.append(ndcg5)
            
            result = {
                "question": q_text,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "precision_at_5": p5,
                "mrr": mrr,
                "ndcg_at_5": ndcg5,
                "num_retrieved": metrics.get("num_retrieved", 10),
                "num_relevant": metrics.get("num_relevant", 0),
            }
            results.append(result)
        else:
            logger.warning(f"No pre-computed results for: {q_text}")
    
    # Aggregate metrics
    def mean(values):
        return sum(values) / len(values) if values else 0.0
    
    baseline_metrics = {
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_config": "hybrid (dense + sparse RRF)",
        "test_set_size": len(test_questions),
        "evaluated": len(results),
        "recall_at_5": mean(recall_at_5),
        "recall_at_10": mean(recall_at_10),
        "precision_at_5": mean(precision_at_5),
        "mrr": mean(mrr_scores),
        "ndcg_at_5": mean(ndcg_at_5),
    }
    
    # Save results
    import os
    os.makedirs("results/baseline", exist_ok=True)
    
    with open("results/baseline/baseline_retrieval_metrics.json", "w") as f:
        json.dump({
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "baseline_metrics": baseline_metrics,
            "per_question": results,
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BASELINE METRICS SUMMARY (Locked Test Set)")
    print("=" * 70)
    print(f"Embedding Model: {baseline_metrics['embedding_model']}")
    print(f"Retrieval Config: {baseline_metrics['retrieval_config']}")
    print(f"Test Set Size: {baseline_metrics['test_set_size']}")
    print(f"Evaluated: {baseline_metrics['evaluated']}")
    print(f"\nMetrics:")
    print(f"  Recall@5:    {baseline_metrics['recall_at_5']:.3f}")
    print(f"  Recall@10:   {baseline_metrics['recall_at_10']:.3f}")
    print(f"  Precision@5: {baseline_metrics['precision_at_5']:.3f}")
    print(f"  MRR:         {baseline_metrics['mrr']:.3f}")
    print(f"  NDCG@5:      {baseline_metrics['ndcg_at_5']:.3f}")
    print(f"\n✓ Results saved: results/baseline/baseline_retrieval_metrics.json")
    print(f"\nNext Step:")
    print(f"  1. Install biomedical embeddings (BioLORD-2023, BioBERT, PubMedBERT)")
    print(f"  2. Re-compute retrieval metrics")
    print(f"  3. Compare against this baseline")

if __name__ == "__main__":
    main()

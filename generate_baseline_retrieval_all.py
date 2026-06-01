"""
generate_baseline_retrieval_all.py
==================================
Generate retrieval results for ALL 221 locked test questions.

This creates the baseline metrics needed for valid evaluation.
Uses existing vectorstore (no re-embedding needed).

Output: dense_results_baseline.json with metrics for all 221 questions
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.ingestion import load_medquad_dataset
from rag.vectorstore import BiomedicalVectorStore
from rag.hybrid_retriever import HybridRetriever
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
    print("GENERATING BASELINE RETRIEVAL RESULTS (all-MiniLM-L6-v2)")
    print("For all 221 locked test questions")
    print("=" * 70)
    
    # Load locked test set
    logger.info("Loading locked test set...")
    test_questions = load_test_set("data/test_set.jsonl")
    logger.info(f"Loaded {len(test_questions)} test questions")
    
    # Initialize retrieval system
    logger.info("\nInitializing retrieval system...")
    start_time = time.time()
    try:
        documents = load_medquad_dataset("data/raw")
        logger.info(f"  Loaded {len(documents)} documents in {time.time()-start_time:.1f}s")
        
        vectorstore = BiomedicalVectorStore()
        vectorstore.add_documents(documents)
        retriever = HybridRetriever(vectorstore=vectorstore, documents=documents)
        logger.info("  ✓ Retrieval system ready\n")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return
    
    # Evaluate all questions
    print("=" * 70)
    print(f"EVALUATING {len(test_questions)} TEST QUESTIONS")
    print("=" * 70 + "\n")
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, question in enumerate(test_questions, 1):
        q_text = question["question"]
        doc_id = question.get("doc_id", -1)
        
        try:
            # Retrieve
            retrieval_result = retriever.retrieve(q_text, k=20)
            
            # Extract retrieved IDs from dense results (fused results don't have metadata)
            chunks = retrieval_result.get("dense_results", [])
            retrieved_ids = [chunk.metadata.get("doc_id", -1) for chunk in chunks]
            
            # Compute metrics
            relevant_ids = {doc_id}
            
            r5 = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k=5)
            r10 = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k=10)
            p5 = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k=5)
            mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg5 = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k=5)
            accuracy = RetrievalMetrics.retrieval_accuracy(retrieved_ids, relevant_ids)
            
            result = {
                "query": q_text,
                "category": question.get("category", "unknown"),
                "doc_id": doc_id,
                "metrics": {
                    "query": q_text,
                    "num_retrieved": len(retrieved_ids),
                    "num_relevant": 1,  # One source document
                    "recall@5": r5,
                    "recall@10": r10,
                    "precision@5": p5,
                    "precision@10": RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k=10),
                    "mrr": mrr,
                    "ndcg@5": ndcg5,
                    "ndcg@10": RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k=10),
                    "accuracy": accuracy,
                    "method": "dense+sparse_rrf"
                }
            }
            results.append(result)
            success_count += 1
            
            # Progress update
            if i % 50 == 0:
                logger.info(f"[{i}/{len(test_questions)}] Completed {success_count} questions")
        
        except Exception as e:
            logger.error(f"Error on question {i}: {q_text[:50]}... → {e}")
            error_count += 1
            continue
    
    # Aggregate metrics
    def mean(values):
        return sum(values) / len(values) if values else 0.0
    
    recall_at_5 = [r["metrics"]["recall@5"] for r in results]
    recall_at_10 = [r["metrics"]["recall@10"] for r in results]
    precision_at_5 = [r["metrics"]["precision@5"] for r in results]
    mrr_scores = [r["metrics"]["mrr"] for r in results]
    ndcg_at_5 = [r["metrics"]["ndcg@5"] for r in results]
    
    baseline_summary = {
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_config": "hybrid (dense + sparse RRF)",
        "test_set_size": len(test_questions),
        "evaluated": len(results),
        "errors": error_count,
        "recall_at_5": mean(recall_at_5),
        "recall_at_10": mean(recall_at_10),
        "precision_at_5": mean(precision_at_5),
        "mrr": mean(mrr_scores),
        "ndcg_at_5": mean(ndcg_at_5),
    }
    
    # Save comprehensive results
    output_file = "dense_results_baseline.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Saved {len(results)} results to {output_file}")
    
    # Save summary
    with open("results/baseline/baseline_full_retrieval_metrics.json", "w") as f:
        json.dump({
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "summary": baseline_summary,
            "per_question": results,
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BASELINE RETRIEVAL COMPLETE")
    print("=" * 70)
    print(f"Test Set Size:      {baseline_summary['test_set_size']}")
    print(f"Evaluated:          {baseline_summary['evaluated']}")
    print(f"Errors:             {baseline_summary['errors']}")
    print(f"Coverage:           {baseline_summary['evaluated']/baseline_summary['test_set_size']*100:.1f}%")
    print(f"\nBaseline Metrics (all-MiniLM-L6-v2):")
    print(f"  Recall@5:    {baseline_summary['recall_at_5']:.3f}")
    print(f"  Recall@10:   {baseline_summary['recall_at_10']:.3f}")
    print(f"  Precision@5: {baseline_summary['precision_at_5']:.3f}")
    print(f"  MRR:         {baseline_summary['mrr']:.3f}")
    print(f"  NDCG@5:      {baseline_summary['ndcg_at_5']:.3f}")
    print(f"\n✓ Results saved:")
    print(f"  - {output_file}")
    print(f"  - results/baseline/baseline_full_retrieval_metrics.json")
    print(f"\nNext: Swap embeddings and re-run to get optimized metrics")

if __name__ == "__main__":
    main()

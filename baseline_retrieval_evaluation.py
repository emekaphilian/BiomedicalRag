"""
baseline_retrieval_evaluation.py
================================
Run retrieval evaluation on LOCKED test set.

Baseline uses: all-MiniLM-L6-v2 embeddings
Configurations: Dense, Sparse (BM25), RRF

This establishes the metrics that retrieval optimization will be measured against.
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
from rag.retrieval_metrics import RetrievalMetrics, RetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_test_set(filepath):
    """Load test set from JSONL."""
    questions = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def load_corpus(filepath):
    """Load indexed corpus."""
    corpus = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus

def main():
    print("=" * 70)
    print("BASELINE RETRIEVAL EVALUATION (all-MiniLM-L6-v2)")
    print("=" * 70)
    
    # Load locked test set
    logger.info("Loading locked test set...")
    test_questions = load_test_set("data/test_set.jsonl")
    logger.info(f"Loaded {len(test_questions)} test questions")
    
    # Load corpus
    logger.info("Loading indexed corpus...")
    corpus = load_corpus("data/indexed_corpus.jsonl")
    logger.info(f"Loaded {len(corpus)} corpus documents")
    
    # Build retriever
    logger.info("Building retriever with all-MiniLM-L6-v2...")
    documents = load_medquad_dataset("data/raw")
    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(documents)
    retriever = HybridRetriever(vectorstore=vectorstore, documents=documents)
    logger.info("✓ Retriever initialized\n")
    
    # Evaluate
    print("=" * 70)
    print("EVALUATING RETRIEVAL PERFORMANCE")
    print("=" * 70 + "\n")
    
    evaluator = RetrievalEvaluator()
    results = []
    recall_at_5 = []
    recall_at_10 = []
    precision_at_5 = []
    mrr_scores = []
    ndcg_at_5 = []
    
    for i, question in enumerate(test_questions, 1):
        q_text = question["question"]
        doc_id = question.get("doc_id", -1)
        
        # Retrieve
        try:
            retrieval_result = retriever.retrieve(q_text, k=20)
            retrieved_chunks = retrieval_result.get("results", [])
            retrieved_ids = [chunk.metadata.get("doc_id", -1) for chunk in retrieved_chunks]
            
            # Evaluate - use static methods from RetrievalMetrics
            relevant_ids = {doc_id}  # The source document is relevant
            
            # Compute metrics using static methods
            r5 = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k=5)
            r10 = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k=10)
            p5 = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k=5)
            mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg5 = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k=5)
            
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
                "retrieved_count": len(retrieved_ids),
            }
            results.append(result)
            
            if i % 50 == 0:
                logger.info(f"[{i}/{len(test_questions)}] Evaluated {i} questions")
        
        except Exception as e:
            logger.error(f"Error evaluating '{q_text}': {e}")
            continue
    
    # Aggregate metrics
    def mean(values):
        return sum(values) / len(values) if values else 0.0
    
    baseline_metrics = {
        "embedding_model": "all-MiniLM-L6-v2",
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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baseline_metrics": baseline_metrics,
            "per_question": results,
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BASELINE METRICS SUMMARY")
    print("=" * 70)
    print(f"Embedding Model: all-MiniLM-L6-v2")
    print(f"Test Set Size: {len(test_questions)}")
    print(f"Evaluated: {len(results)}")
    print(f"\nMetrics:")
    print(f"  Recall@5:    {baseline_metrics['recall_at_5']:.3f}")
    print(f"  Recall@10:   {baseline_metrics['recall_at_10']:.3f}")
    print(f"  Precision@5: {baseline_metrics['precision_at_5']:.3f}")
    print(f"  MRR:         {baseline_metrics['mrr']:.3f}")
    print(f"  NDCG@5:      {baseline_metrics['ndcg_at_5']:.3f}")
    print(f"\n✓ Results saved: results/baseline/baseline_retrieval_metrics.json")
    print(f"\nNext: Swap embeddings to BioLORD-2023 and re-measure.")

if __name__ == "__main__":
    main()

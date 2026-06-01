"""
Task 3B: Retrieval Evaluation with Academic Metrics

Measures retrieval quality using standard IR metrics:
- Recall@5, Recall@10
- Precision@5, Precision@10
- Mean Reciprocal Rank (MRR)
- NDCG@5, NDCG@10

This directly aligns with proposal Section 4.3 and addresses the
evaluation framework gap identified in the analysis.
"""

import json
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingestion import load_medquad_dataset
from rag.vectorstore import BiomedicalVectorStore
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker, HybridReranker
from rag.retrieval_metrics import RetrievalEvaluator, RelevanceJudgement

# Same 18 test queries
EVALUATION_QUERIES = {
    "A_FACTOID": [
        "What are the main symptoms of diabetes?",
        "What causes asthma?",
        "What are symptoms of hypertension?",
        "What is melanoma?",
        "What causes anemia?",
    ],
    "B_MECHANISM": [
        "How does insulin regulate blood sugar?",
        "How does Parkinson's disease affect dopamine?",
        "What causes autoimmune disorders?",
        "How does hypertension damage the heart?",
    ],
    "C_MULTIHOP": [
        "How are obesity and type 2 diabetes related?",
        "What complications can untreated hypertension cause?",
        "How does smoking contribute to lung cancer?",
    ],
    "D_AMBIGUOUS": [
        "What is the best cancer treatment?",
        "How do I cure infection?",
        "Why am I always tired?",
    ],
    "E_ADVERSARIAL": [
        "What is the FDA-approved cure for Alzheimer's?",
        "Does vitamin C permanently reverse Parkinson's disease?",
        "Can leukemia be cured naturally?",
    ],
}


def main():
    """Run retrieval evaluation with academic metrics"""

    print("\n" + "=" * 80)
    print("TASK 3B: RETRIEVAL EVALUATION WITH ACADEMIC METRICS")
    print("=" * 80 + "\n")

    # Setup
    print("[SETUP] Loading biomedical dataset...")
    start_time = time.time()
    documents = load_medquad_dataset(dataset_path="data/raw")
    print(f"[OK] Loaded {len(documents)} documents in {time.time() - start_time:.1f}s\n")

    print("[SETUP] Building retrieval indexes...")
    start_time = time.time()
    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(documents)
    retriever = HybridRetriever(vectorstore=vectorstore, documents=documents)
    print(f"[OK] Indexes built in {time.time() - start_time:.1f}s\n")

    print("[SETUP] Loading reranker...")
    start_time = time.time()
    reranker = CrossEncoderReranker()
    print(f"[OK] Cross-encoder loaded in {time.time() - start_time:.1f}s\n")

    # Evaluation
    print("=" * 80)
    print("RUNNING RETRIEVAL EVALUATION")
    print("=" * 80 + "\n")

    evaluator = RetrievalEvaluator()
    all_results = []
    category_stats = {cat: {} for cat in EVALUATION_QUERIES}

    for category, queries in EVALUATION_QUERIES.items():
        print(f"[{category}] Evaluating {len(queries)} queries...\n")

        for idx, query in enumerate(queries, 1):
            # Retrieve
            retrieval_result = retriever.retrieve(query, k=20)
            fused_results = retrieval_result["fused_results"]

            # Extract documents from fused results
            retrieved_docs = [content for content, _ in fused_results]

            # Rerank with cross-encoder
            reranked = reranker.rerank_documents(query, retrieved_docs, k=10)
            reranked_docs = [doc for doc, _ in reranked]

            # Get relevant keywords
            relevant_keywords = RelevanceJudgement.get_relevant_keywords(query)

            # Evaluate retrieval (use fused results as baseline)
            result_before = evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs[:10],
                metadata={
                    "category": category,
                    "method": "fusion",
                    "num_relevant_keywords": len(relevant_keywords),
                },
            )

            # Evaluate after reranking
            result_after = evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=reranked_docs,
                metadata={
                    "category": category,
                    "method": "reranked",
                    "num_relevant_keywords": len(relevant_keywords),
                },
            )

            all_results.append({
                "query": query,
                "category": category,
                "before_reranking": result_before,
                "after_reranking": result_after,
                "improvement": {
                    "recall@5": result_after["recall@5"] - result_before["recall@5"],
                    "precision@5": result_after["precision@5"] - result_before["precision@5"],
                    "mrr": result_after["mrr"] - result_before["mrr"],
                    "ndcg@5": result_after["ndcg@5"] - result_before["ndcg@5"],
                },
            })

            # Print progress
            print(
                f"  [{idx}/{len(queries)}] {query[:50]}...\n"
                f"    Before: Recall@5={result_before['recall@5']:.2f}, "
                f"Precision@5={result_before['precision@5']:.2f}\n"
                f"    After:  Recall@5={result_after['recall@5']:.2f}, "
                f"Precision@5={result_after['precision@5']:.2f}\n"
            )

            category_stats[category][query] = all_results[-1]

    # Save results
    print(f"\n[SAVE] Writing results to retrieval_evaluation_results.jsonl...")
    with open("retrieval_evaluation_results.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print("[OK] Results saved\n")

    # Summary statistics
    print("=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80 + "\n")

    summary = evaluator.get_summary()

    print("Before Reranking (Fusion-only):")
    for metric in [
        "recall@5_mean",
        "precision@5_mean",
        "mrr_mean",
        "ndcg@5_mean",
    ]:
        if metric.replace("_mean", "_before") in summary or metric in summary:
            key = metric.replace("_mean", "_before") if metric.replace("_mean", "_before") in summary else metric
            if key in summary:
                print(f"  {metric:20} {summary[key]:.3f}")

    print("\nRetrieval Metrics (After Reranking):")
    for metric, value in sorted(summary.items()):
        if "_mean" in metric:
            print(f"  {metric:20} {value:.3f}")

    # Category breakdown
    print("\nPerformance by Category:")
    for category in sorted(EVALUATION_QUERIES.keys()):
        queries_in_cat = list(category_stats[category].keys())
        if queries_in_cat:
            improvements = [
                category_stats[category][q]["improvement"]["recall@5"]
                for q in queries_in_cat
            ]
            avg_improvement = sum(improvements) / len(improvements)
            print(
                f"  {category:15} Recall@5 Improvement: {avg_improvement:+.3f}"
            )

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80 + "\n")

    total_queries = len(all_results)
    improved_queries = sum(
        1 for r in all_results if r["improvement"]["recall@5"] > 0
    )
    degraded_queries = sum(
        1 for r in all_results if r["improvement"]["recall@5"] < 0
    )

    print(
        f"Reranking improved recall@5 on {improved_queries}/{total_queries} "
        f"queries ({improved_queries / total_queries * 100:.1f}%)"
    )
    print(
        f"Reranking degraded recall@5 on {degraded_queries}/{total_queries} "
        f"queries ({degraded_queries / total_queries * 100:.1f}%)"
    )

    avg_recall_before = sum(r["before_reranking"]["recall@5"] for r in all_results) / len(all_results)
    avg_recall_after = sum(r["after_reranking"]["recall@5"] for r in all_results) / len(all_results)
    print(
        f"\nAverage Recall@5: {avg_recall_before:.3f} → {avg_recall_after:.3f} "
        f"({(avg_recall_after - avg_recall_before) * 100:+.1f} percentage points)"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

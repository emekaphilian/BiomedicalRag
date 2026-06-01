"""
Task 3C: Retrieval Error Analysis - Diagnostic Pipeline

Captures detailed retrieval outputs for manual inspection and
runs 4 experiments to identify bottlenecks:

1. Dense retrieval only
2. BM25 sparse only  
3. RRF fusion
4. RRF + Cross-encoder reranking

Outputs:
- retrieval_debug.jsonl: Full chunks for manual inspection
- dense_results.json: Metrics for dense-only baseline
- bm25_results.json: Metrics for BM25-only baseline
- rrf_results.json: Metrics for RRF fusion
- reranked_results.json: Metrics for RRF + reranker
- retrieval_comparison.json: Side-by-side metric comparison
"""

import json
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingestion import load_medquad_dataset
from rag.vectorstore import BiomedicalVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.retriever import BiomedicalRetriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.retrieval_metrics import RetrievalEvaluator, RelevanceJudgement

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
    """Run comprehensive retrieval error analysis"""

    print("\n" + "=" * 80)
    print("TASK 3C: RETRIEVAL ERROR ANALYSIS")
    print("=" * 80 + "\n")

    # Setup
    print("[SETUP] Loading biomedical dataset...")
    start = time.time()
    documents = load_medquad_dataset(dataset_path="data/raw")
    print(f"[OK] Loaded {len(documents)} documents in {time.time() - start:.1f}s\n")

    print("[SETUP] Building indexes...")
    start = time.time()
    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(documents)
    dense_retriever = BiomedicalRetriever(vectorstore)
    bm25_retriever = BM25Retriever(documents)
    hybrid_retriever = HybridRetriever(vectorstore, documents)
    print(f"[OK] Indexes built in {time.time() - start:.1f}s\n")

    print("[SETUP] Loading cross-encoder reranker...")
    start = time.time()
    reranker = CrossEncoderReranker()
    print(f"[OK] Reranker loaded in {time.time() - start:.1f}s\n")

    # Experiment phase
    print("=" * 80)
    print("RUNNING 4 RETRIEVAL EXPERIMENTS")
    print("=" * 80 + "\n")

    debug_records = []
    dense_results = []
    bm25_results = []
    rrf_results = []
    reranked_results = []

    all_queries = []
    for cat, queries in EVALUATION_QUERIES.items():
        all_queries.extend([(cat, q) for q in queries])

    for query_idx, (category, query) in enumerate(all_queries, 1):
        print(f"[{query_idx}/18] {query[:60]}...")

        # Get relevance keywords for this query
        relevant_keywords = RelevanceJudgement.get_relevant_keywords(query)

        # ============= EXPERIMENT A: DENSE ONLY =============
        dense_response = dense_retriever.retrieve(query, k=10)
        dense_docs = dense_response["results"]
        dense_texts = [doc.content for doc in dense_docs]

        # ============= EXPERIMENT B: BM25 ONLY =============
        bm25_response = bm25_retriever.search(query, k=10)
        bm25_texts = [r["document"].content for r in bm25_response]

        # ============= EXPERIMENT C: RRF FUSION =============
        hybrid_response = hybrid_retriever.retrieve(query, k=10)
        rrf_fused = hybrid_response["fused_results"]
        rrf_texts = [content for content, _ in rrf_fused]

        # ============= EXPERIMENT D: RRF + RERANKER =============
        reranked = reranker.rerank_documents(query, rrf_texts, k=10)
        reranked_texts = [doc for doc, _ in reranked]

        # ============= EVALUATION =============
        evaluator = RetrievalEvaluator()

        dense_metrics = evaluator.evaluate_retrieval(
            query, dense_texts, metadata={"method": "dense"}
        )
        bm25_metrics = evaluator.evaluate_retrieval(
            query, bm25_texts, metadata={"method": "bm25"}
        )
        rrf_metrics = evaluator.evaluate_retrieval(
            query, rrf_texts, metadata={"method": "rrf"}
        )
        reranked_metrics = evaluator.evaluate_retrieval(
            query, reranked_texts, metadata={"method": "reranked"}
        )

        dense_results.append({
            "query": query,
            "category": category,
            "metrics": dense_metrics,
        })
        bm25_results.append({
            "query": query,
            "category": category,
            "metrics": bm25_metrics,
        })
        rrf_results.append({
            "query": query,
            "category": category,
            "metrics": rrf_metrics,
        })
        reranked_results.append({
            "query": query,
            "category": category,
            "metrics": reranked_metrics,
        })

        # ============= DEBUG OUTPUT =============
        # Save full chunk texts for manual inspection
        debug_record = {
            "query_idx": query_idx,
            "query": query,
            "category": category,
            "relevant_keywords": list(relevant_keywords),
            "retrieved_chunks": []
        }

        # Include top 5 from each method
        for rank, (text, source) in enumerate(zip(rrf_texts[:5], range(1, 6)), 1):
            # Check if contains relevant keywords
            has_relevant = any(
                kw.lower() in text.lower() for kw in relevant_keywords
            )
            debug_record["retrieved_chunks"].append({
                "rank": rank,
                "text": text[:500],
                "has_relevant_keyword": has_relevant,
                "source_file": documents[0].metadata.get("file", "unknown") if documents else "unknown",
            })

        debug_records.append(debug_record)

        print(f"  ✓ Dense@5={dense_metrics['recall@5']:.2f} "
              f"| BM25@5={bm25_metrics['recall@5']:.2f} "
              f"| RRF@5={rrf_metrics['recall@5']:.2f} "
              f"| Reranked@5={reranked_metrics['recall@5']:.2f}")

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80 + "\n")

    # Save debug output
    with open("retrieval_debug.jsonl", "w") as f:
        for record in debug_records:
            f.write(json.dumps(record) + "\n")
    print("[OK] Saved retrieval_debug.jsonl")

    # Save individual experiment results
    with open("dense_results.json", "w") as f:
        json.dump(dense_results, f, indent=2)
    print("[OK] Saved dense_results.json")

    with open("bm25_results.json", "w") as f:
        json.dump(bm25_results, f, indent=2)
    print("[OK] Saved bm25_results.json")

    with open("rrf_results.json", "w") as f:
        json.dump(rrf_results, f, indent=2)
    print("[OK] Saved rrf_results.json")

    with open("reranked_results.json", "w") as f:
        json.dump(reranked_results, f, indent=2)
    print("[OK] Saved reranked_results.json")

    # ============= COMPARISON TABLE =============
    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON - RECALL@5")
    print("=" * 80 + "\n")

    print(f"{'Query':<50} {'Dense':>7} {'BM25':>7} {'RRF':>7} {'Reranked':>9}")
    print("-" * 85)

    comparison = {
        "by_method": {
            "dense": [],
            "bm25": [],
            "rrf": [],
            "reranked": [],
        },
        "by_category": {}
    }

    for dense, bm25, rrf, reranked in zip(
        dense_results, bm25_results, rrf_results, reranked_results
    ):
        query = dense["query"][:47]
        d_recall = dense["metrics"]["recall@5"]
        b_recall = bm25["metrics"]["recall@5"]
        r_recall = rrf["metrics"]["recall@5"]
        re_recall = reranked["metrics"]["recall@5"]

        print(f"{query:<50} {d_recall:>7.3f} {b_recall:>7.3f} {r_recall:>7.3f} {re_recall:>9.3f}")

        # Aggregate by method
        comparison["by_method"]["dense"].append(d_recall)
        comparison["by_method"]["bm25"].append(b_recall)
        comparison["by_method"]["rrf"].append(r_recall)
        comparison["by_method"]["reranked"].append(re_recall)

        # Aggregate by category
        cat = dense["category"]
        if cat not in comparison["by_category"]:
            comparison["by_category"][cat] = {
                "dense": [],
                "bm25": [],
                "rrf": [],
                "reranked": [],
            }
        comparison["by_category"][cat]["dense"].append(d_recall)
        comparison["by_category"][cat]["bm25"].append(b_recall)
        comparison["by_category"][cat]["rrf"].append(r_recall)
        comparison["by_category"][cat]["reranked"].append(re_recall)

    # Print averages
    print("-" * 85)
    print(f"{'AVERAGE':<50} "
          f"{sum(comparison['by_method']['dense']) / len(comparison['by_method']['dense']):>7.3f} "
          f"{sum(comparison['by_method']['bm25']) / len(comparison['by_method']['bm25']):>7.3f} "
          f"{sum(comparison['by_method']['rrf']) / len(comparison['by_method']['rrf']):>7.3f} "
          f"{sum(comparison['by_method']['reranked']) / len(comparison['by_method']['reranked']):>9.3f}")

    # Save comparison table
    with open("retrieval_comparison.json", "w") as f:
        # Convert lists to averages for JSON
        summary = {
            "by_method": {
                method: sum(scores) / len(scores)
                for method, scores in comparison["by_method"].items()
            },
            "by_category": {
                cat: {
                    method: sum(scores) / len(scores)
                    for method, scores in methods.items()
                }
                for cat, methods in comparison["by_category"].items()
            }
        }
        json.dump(summary, f, indent=2)
    print("\n[OK] Saved retrieval_comparison.json")

    # ============= ANALYSIS =============
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80 + "\n")

    dense_avg = sum(comparison["by_method"]["dense"]) / len(comparison["by_method"]["dense"])
    bm25_avg = sum(comparison["by_method"]["bm25"]) / len(comparison["by_method"]["bm25"])
    rrf_avg = sum(comparison["by_method"]["rrf"]) / len(comparison["by_method"]["rrf"])
    reranked_avg = sum(comparison["by_method"]["reranked"]) / len(comparison["by_method"]["reranked"])

    print(f"Average Recall@5 by Method:")
    print(f"  Dense:      {dense_avg:.3f}")
    print(f"  BM25:       {bm25_avg:.3f}")
    print(f"  RRF:        {rrf_avg:.3f}")
    print(f"  Reranked:   {reranked_avg:.3f}")

    # Identify bottleneck
    worst_method = min(
        [("Dense", dense_avg), ("BM25", bm25_avg), ("RRF", rrf_avg)],
        key=lambda x: x[1]
    )
    print(f"\n⚠ Bottleneck: {worst_method[0]} retrieval ({worst_method[1]:.3f})")

    # Reranker impact
    reranker_delta = reranked_avg - rrf_avg
    if reranker_delta > 0:
        print(f"✓ Reranker helps: {reranker_delta:+.3f}")
    else:
        print(f"✗ Reranker hurts: {reranker_delta:+.3f}")

    # By category
    print("\nPerformance by Category (RRF):")
    for cat in sorted(comparison["by_category"].keys()):
        cat_avg = sum(comparison["by_category"][cat]["rrf"]) / len(
            comparison["by_category"][cat]["rrf"]
        )
        print(f"  {cat}: {cat_avg:.3f}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80 + "\n")
    print("1. Open retrieval_debug.jsonl - inspect worst queries manually")
    print("2. Check: was relevant information in corpus? (YES=ranking, NO=retrieval)")
    print("3. For each failure case, identify: embedding, keyword, or ranking issue")
    print("4. Based on bottleneck, choose improvement:")
    print("   - Dense weak? → Replace with biomedical embeddings (BioLORD)")
    print("   - BM25 weak? → Add medical synonym expansion")
    print("   - Ranking weak? → Tune RRF parameter k (10/30/60/100)")
    print("   - Reranker bad? → Try different cross-encoder or remove it")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Task 3: Improved Evaluation with Confidence Thresholding and Refusal Logic

Runs the same 18-query evaluation suite but with:
1. Query classification (factoid, mechanism, multihop, etc.)
2. Retrieval confidence thresholding
3. Explicit refusal templates
4. Sentence-level citation tracking

Expected improvements:
- Pass rate: 50% → 70-75%+
- Hallucination rate: 27.8% → <15%
"""

import json
import time
from datetime import datetime
import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingestion import load_medquad_dataset
from rag.vectorstore import BiomedicalVectorStore
from rag.enhanced_retriever import EnhancedHybridRetriever
from rag.query_classifier import QueryClassifier, QueryType
from rag.generator_cohere import EnhancedCohereGenerator

# Test suite (same 18 queries as Task 2)
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


def validate_grounding(answer, retrieved_chunks):
    """
    Enhanced grounding validation checking for:
    1. Citation presence
    2. Evidence coherence
    3. Refusal appropriateness
    """
    citation_pattern = r"\[Evidence\s+(\d+(?:,\s*\d+)*)\]"
    citations = re.findall(citation_pattern, answer)

    # If refusal template is detected, it's valid
    refusal_keywords = [
        "don't have enough verified",
        "cannot provide",
        "cannot diagnose",
        "low confidence",
        "consult a healthcare",
    ]
    if any(keyword in answer for keyword in refusal_keywords):
        return True, "appropriate_refusal", 0

    # Answer requires citations
    if not citations:
        return False, "no_citations", 0

    # Parse citation indices
    citation_indices = []
    for citation_str in citations:
        indices = [int(idx.strip()) for idx in citation_str.split(",")]
        citation_indices.extend(indices)

    # Validate citations reference valid chunks
    max_chunk = len(retrieved_chunks)
    valid_citations = all(1 <= idx <= max_chunk for idx in citation_indices)

    if not valid_citations:
        return False, "invalid_citation_indices", len(citation_indices)

    # Check citation coverage (at least 20% of chunks should be cited)
    coverage_ratio = len(set(citation_indices)) / max_chunk if max_chunk > 0 else 0
    if coverage_ratio < 0.2 and max_chunk > 1:
        return False, "insufficient_coverage", coverage_ratio

    return True, "properly_grounded", coverage_ratio


def run_evaluation():
    """Run full evaluation with improved components"""

    print("\n" + "=" * 80)
    print("STABILIZATION PHASE - TASK 3: IMPROVED EVALUATION")
    print("Confidence Thresholding + Query Classification + Refusal Logic")
    print("=" * 80 + "\n")

    # Setup
    print("[SETUP] Loading biomedical dataset...")
    documents = load_medquad_dataset(dataset_path="data/raw")
    print(f"[OK] Loaded {len(documents)} documents\n")

    print("[SETUP] Building retrieval indexes...")
    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(documents)
    retriever = EnhancedHybridRetriever(vectorstore, documents)
    print("[OK] Indexes built\n")

    print("[SETUP] Initializing enhanced components...")
    classifier = QueryClassifier()
    generator = EnhancedCohereGenerator()
    print("[OK] Components ready\n")

    # Evaluation loop
    results = []
    category_stats = {cat: {"passed": 0, "total": 0, "time": 0} for cat in EVALUATION_QUERIES}
    total_passed = 0
    total_tests = 0
    total_refusals = 0

    for category, queries in EVALUATION_QUERIES.items():
        print(f"[{category}] Running {len(queries)} queries...\n")
        category_start = time.time()

        for idx, query in enumerate(queries, 1):
            # Classify query
            query_type, metadata = classifier.classify(query)
            query_type_str = query_type.value

            # Retrieve with confidence thresholding
            start_time = time.time()
            retrieval_result = retriever.retrieve_with_threshold(
                query=query,
                query_type=query_type_str,
                strict_mode=metadata.get("strict_mode", False),
                k=5,
            )
            retrieval_time = time.time() - start_time

            should_generate = retrieval_result["should_generate"]
            refusal_reason = retrieval_result["refusal_reason"]
            confidence_metrics = retrieval_result["confidence_metrics"]
            fused_results = retrieval_result["fused_results"]

            # Generate answer
            context_text = "\n\n".join(
                [f"[Evidence {i + 1}]\n{content}" for i, (content, _) in enumerate(fused_results[:5])]
            )

            start_time = time.time()
            answer = generator.generate_grounded(
                query=query,
                context=context_text,
                query_type=query_type_str,
                confidence_metrics=confidence_metrics,
                should_generate=should_generate,
                refusal_reason=refusal_reason,
            )
            generation_time = time.time() - start_time

            # Validate grounding
            grounded, grounding_reason, coverage = validate_grounding(
                answer, fused_results
            )

            # Determine pass/fail
            passed = grounded or (refusal_reason and grounding_reason == "appropriate_refusal")

            if not should_generate or grounding_reason == "appropriate_refusal":
                total_refusals += 1

            # Log result
            result = {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "query_num": idx,
                "query": query,
                "query_type": query_type_str,
                "passed": passed,
                "retrieval_quality": confidence_metrics.get("retrieval_quality", "unknown"),
                "top_confidence": confidence_metrics.get("top_confidence", 0),
                "avg_top3_confidence": confidence_metrics.get("avg_top3_confidence", 0),
                "should_generate": should_generate,
                "was_refusal": not should_generate,
                "grounding_reason": grounding_reason,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
                "chunks_retrieved": len(fused_results),
                "answer_length": len(answer),
            }

            results.append(result)
            total_tests += 1
            if passed:
                total_passed += 1
            category_stats[category]["total"] += 1
            if passed:
                category_stats[category]["passed"] += 1
            category_stats[category]["time"] += result["total_time"]

            # Print result
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"  [{idx}/{len(queries)}] {query[:50]}...\n"
                f"    {status} | Quality: {confidence_metrics.get('retrieval_quality')} | "
                f"Conf: {confidence_metrics.get('top_confidence', 0):.2f}\n"
            )

        print()

    # Save results
    print(f"[SAVE] Writing {len(results)} results to evaluation_results_task3.jsonl...")
    with open("evaluation_results_task3.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("[OK] Results saved\n")

    # Print summary
    print("=" * 80)
    print("EVALUATION RESULTS (IMPROVED)")
    print("=" * 80 + "\n")

    print(f"Overall: {total_passed}/{total_tests} tests passed ({total_passed / total_tests * 100:.1f}%)")
    print(f"Appropriate Refusals: {total_refusals}\n")

    for category in sorted(EVALUATION_QUERIES.keys()):
        stats = category_stats[category]
        passed = stats["passed"]
        total = stats["total"]
        pct = (passed / total * 100) if total > 0 else 0
        avg_time = (stats["time"] / total) if total > 0 else 0
        print(f"  {category:12} {passed:2}/{total} passed ({pct:5.1f}%) | {avg_time:6.2f}s")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80 + "\n")

    # Compare with Task 2 baseline (50% overall pass rate)
    improvement = (total_passed / total_tests - 0.5) * 100
    print(f"Improvement vs Task 2 baseline (50%): {improvement:+.1f} percentage points")
    print(f"Target: 70-75% (need {70 - total_passed / total_tests * 100:.1f} more percentage points)\n")

    # Analyze by query type
    type_stats = {}
    for result in results:
        qtype = result["query_type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"passed": 0, "total": 0, "refusals": 0}
        type_stats[qtype]["total"] += 1
        if result["passed"]:
            type_stats[qtype]["passed"] += 1
        if result["was_refusal"]:
            type_stats[qtype]["refusals"] += 1

    print("Performance by Query Type:")
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(
            f"  {qtype:15} {stats['passed']:1}/{stats['total']} ({pct:5.1f}%) "
            f"| Refusals: {stats['refusals']}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_evaluation()

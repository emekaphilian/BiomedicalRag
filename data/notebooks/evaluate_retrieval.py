"""
evaluate_retrieval.py
=====================
Tier 1: Retrieval Evaluation
Computes Recall@5, Recall@10, Precision@5, MRR, and latency p95
across three retriever configurations:
  - dense   : FAISS dense retrieval (your existing setup)
  - sparse  : BM25 sparse retrieval
  - hybrid  : Reciprocal Rank Fusion of dense + sparse

Usage
-----
    python evaluate_retrieval.py \
        --test_set      data/test_set.jsonl \
        --retriever_url http://localhost:8000   \
        --output_dir    results/

The script calls your existing retriever via HTTP.  Adapt the
`query_retriever()` function if your interface differs.

Proposal targets (fail thresholds in parentheses)
--------------------------------------------------
    Recall@5     ≥ 0.75  (< 0.60)
    Recall@10    ≥ 0.85  (< 0.70)
    Precision@5  ≥ 0.60  (< 0.45)
    MRR          ≥ 0.65  (< 0.50)
    Latency p95  < 300ms (> 800ms)
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Targets & thresholds from proposal (Section 4.3)
# ---------------------------------------------------------------------------
TARGETS = {
    "recall_at_5":   {"target": 0.75, "fail": 0.60},
    "recall_at_10":  {"target": 0.85, "fail": 0.70},
    "precision_at_5": {"target": 0.60, "fail": 0.45},
    "mrr":           {"target": 0.65, "fail": 0.50},
    "latency_p95_ms": {"target": 300,  "fail": 800},
}

RETRIEVER_CONFIGS = ["dense", "sparse", "hybrid"]


# ---------------------------------------------------------------------------
# HTTP interface to your existing retriever
# ---------------------------------------------------------------------------

def query_retriever(
    question: str,
    config: str,
    retriever_url: str,
    k: int = 10,
    timeout: float = 10.0,
) -> tuple[list[dict], float]:
    """
    Call the retriever and return (results, latency_ms).

    Expected response JSON:
        {
          "results": [
            {"doc_id": "...", "score": 0.92, "text": "...", "source": "..."},
            ...
          ]
        }

    Adapt this function to match your actual retriever API.
    """
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{retriever_url}/retrieve",
            json={"query": question, "config": config, "k": k},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000
        return data.get("results", []), latency_ms
    except requests.exceptions.RequestException as e:
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"  [WARN] Retriever error for '{question[:40]}…': {e}")
        return [], latency_ms


# ---------------------------------------------------------------------------
# Relevance judgement
# ---------------------------------------------------------------------------

def is_relevant(result: dict, supporting_docs: list[str]) -> bool:
    """
    Returns True if the retrieved result matches any of the supporting docs.

    Matching strategy (in order of strictness):
      1. Exact doc_id match
      2. Source file substring match
      3. Topic/focus keyword overlap with the question answer

    Adapt to your corpus's doc_id scheme.
    """
    if not supporting_docs:
        # No ground truth provided — skip this question for precision/recall
        return False

    doc_id = str(result.get("doc_id", "")).lower()
    source = str(result.get("source", "")).lower()

    for ref in supporting_docs:
        ref_lower = ref.lower()
        if ref_lower in doc_id or ref_lower in source:
            return True
        # Substring match on filename stem
        if Path(ref_lower).stem in doc_id or Path(ref_lower).stem in source:
            return True
    return False


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(results: list[dict], supporting_docs: list[str], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0 (binary recall per query)."""
    if not supporting_docs:
        return float("nan")
    top_k = results[:k]
    return 1.0 if any(is_relevant(r, supporting_docs) for r in top_k) else 0.0


def compute_precision_at_k(results: list[dict], supporting_docs: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if not supporting_docs:
        return float("nan")
    top_k = results[:k]
    if not top_k:
        return 0.0
    n_relevant = sum(1 for r in top_k if is_relevant(r, supporting_docs))
    return n_relevant / len(top_k)


def compute_reciprocal_rank(results: list[dict], supporting_docs: list[str]) -> float:
    """1/rank of the first relevant result (0.0 if none found)."""
    if not supporting_docs:
        return float("nan")
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, supporting_docs):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Per-config evaluation loop
# ---------------------------------------------------------------------------

def evaluate_config(
    test_questions: list[dict],
    config: str,
    retriever_url: str,
    k_max: int = 10,
) -> dict[str, Any]:
    """Run evaluation for a single retriever config. Returns metric dict."""

    recall5_scores   = []
    recall10_scores  = []
    precision5_scores = []
    mrr_scores       = []
    latencies_ms     = []

    n_skipped = 0

    for i, item in enumerate(test_questions):
        question       = item["question"]
        supporting_docs = item.get("supporting_docs", [])

        # Skip questions with no ground truth supporting docs
        if not supporting_docs:
            n_skipped += 1

        results, latency_ms = query_retriever(
            question=question,
            config=config,
            retriever_url=retriever_url,
            k=k_max,
        )
        latencies_ms.append(latency_ms)

        if supporting_docs:
            r5  = compute_recall_at_k(results, supporting_docs, k=5)
            r10 = compute_recall_at_k(results, supporting_docs, k=10)
            p5  = compute_precision_at_k(results, supporting_docs, k=5)
            rr  = compute_reciprocal_rank(results, supporting_docs)

            recall5_scores.append(r5)
            recall10_scores.append(r10)
            precision5_scores.append(p5)
            mrr_scores.append(rr)

        if (i + 1) % 25 == 0:
            print(f"    [{config}] {i+1}/{len(test_questions)} processed …")

    def nanmean(lst):
        arr = [x for x in lst if not (isinstance(x, float) and x != x)]
        return float(np.mean(arr)) if arr else float("nan")

    latencies_arr = np.array(latencies_ms)

    return {
        "config":           config,
        "n_evaluated":      len(test_questions) - n_skipped,
        "n_skipped":        n_skipped,
        "recall_at_5":      nanmean(recall5_scores),
        "recall_at_10":     nanmean(recall10_scores),
        "precision_at_5":   nanmean(precision5_scores),
        "mrr":              nanmean(mrr_scores),
        "latency_mean_ms":  float(np.mean(latencies_arr)),
        "latency_p50_ms":   float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms":   float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms":   float(np.percentile(latencies_arr, 99)),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def pass_fail(value: float, metric: str) -> str:
    if metric not in TARGETS or value != value:  # nan check
        return "N/A"
    t = TARGETS[metric]["target"]
    f = TARGETS[metric]["fail"]
    # For latency, lower is better
    if "latency" in metric:
        if value <= t:
            return "✅ PASS"
        if value >= f:
            return "❌ FAIL"
        return "⚠️  WARN"
    else:
        if value >= t:
            return "✅ PASS"
        if value <= f:
            return "❌ FAIL"
        return "⚠️  WARN"


def print_report(all_results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  TIER 1 RETRIEVAL EVALUATION REPORT")
    print("=" * 70)

    metrics = [
        ("recall_at_5",   "Recall@5"),
        ("recall_at_10",  "Recall@10"),
        ("precision_at_5","Precision@5"),
        ("mrr",           "MRR"),
        ("latency_p95_ms","Latency p95 (ms)"),
    ]

    header = f"{'Metric':<20}" + "".join(f"{r['config']:>14}" for r in all_results)
    print(header)
    print("-" * (20 + 14 * len(all_results)))

    for key, label in metrics:
        row = f"{label:<20}"
        for r in all_results:
            val = r[key]
            pf  = pass_fail(val, key)
            if "latency" in key:
                row += f"{val:>8.1f}ms  "
            else:
                row += f"{val:>8.3f}    "
        print(row)

    print()
    print("Pass/Fail summary:")
    for key, label in metrics:
        for r in all_results:
            val = r[key]
            status = pass_fail(val, key)
            print(f"  [{r['config']:>7}] {label:<20}: {status}")

    print()
    print("Proposal targets:")
    for key, thresh in TARGETS.items():
        direction = "≤" if "latency" in key else "≥"
        print(f"  {key:<20}: target {direction}{thresh['target']}  |  fail threshold {'≥' if 'latency' in key else '≤'}{thresh['fail']}")
    print("=" * 70)


def save_results(all_results: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"retrieval_eval_{ts}.json")
    with open(path, "w") as f:
        json.dump({
            "timestamp": ts,
            "targets":   TARGETS,
            "results":   all_results,
        }, f, indent=2)
    print(f"\n  Results saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier 1: Retrieval Evaluation")
    parser.add_argument("--test_set",      default="data/test_set.jsonl",
                        help="Path to locked test set JSONL")
    parser.add_argument("--retriever_url", default="http://localhost:8000",
                        help="Base URL of your retriever service")
    parser.add_argument("--output_dir",    default="results/",
                        help="Directory to save results JSON")
    parser.add_argument("--configs",       nargs="+", default=RETRIEVER_CONFIGS,
                        choices=RETRIEVER_CONFIGS,
                        help="Which retriever configurations to evaluate")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Limit to first N questions (for quick testing)")
    args = parser.parse_args()

    # --- Verify test set integrity before running ---
    from verify_test_set import verify
    lock_path = args.test_set.replace(".jsonl", ".lock")
    if not verify(args.test_set, lock_path):
        print("\n[ABORT] Test set failed integrity check. Evaluation cancelled.")
        raise SystemExit(1)

    # --- Load test set ---
    with open(args.test_set) as f:
        test_questions = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        test_questions = test_questions[: args.limit]
        print(f"  [DEV MODE] Limited to first {args.limit} questions")

    print(f"\n  Evaluating {len(test_questions)} questions across configs: {args.configs}")
    print(f"  Retriever URL : {args.retriever_url}\n")

    # --- Evaluate each config ---
    all_results = []
    for config in args.configs:
        print(f"\n  ── Config: {config} ──")
        result = evaluate_config(
            test_questions=test_questions,
            config=config,
            retriever_url=args.retriever_url,
        )
        all_results.append(result)

    # --- Report ---
    print_report(all_results)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()

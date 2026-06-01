"""
run_evaluation.py
=================
Master evaluation runner.

Runs in order:
    1. verify_test_set.py        — abort if test set tampered
    2. evaluate_retrieval.py     — Tier 1 metrics
    3. evaluate_generation.py    — Tier 2 metrics
    4. Generates combined HTML report

Usage
-----
    python run_evaluation.py \
        --test_set      data/test_set.jsonl \
        --retriever_url http://localhost:8000 \
        --rag_url       http://localhost:8000 \
        --cohere_key    $COHERE_API_KEY \
        --output_dir    results/

    # Retrieval only (fast):
    python run_evaluation.py --tier retrieval ...

    # Generation only:
    python run_evaluation.py --tier generation ...
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

PASS_COLORS = {
    "✅ PASS": "#2d6a2d",
    "⚠️  WARN": "#7a5c00",
    "❌ FAIL": "#8b0000",
    "N/A":     "#555",
}

def color_cell(value: str, status: str) -> str:
    color = PASS_COLORS.get(status, "#333")
    return f'<td style="color:{color};font-weight:bold">{value}</td>'


def generate_html_report(
    retrieval_results: list[dict] | None,
    generation_results: list[dict] | None,
    output_dir: str,
    timestamp: str,
) -> str:
    from evaluate_retrieval   import TARGETS as R_TARGETS, pass_fail as r_pf
    from evaluate_generation  import TARGETS as G_TARGETS, pass_fail as g_pf

    def fmt(v):
        if isinstance(v, float) and v != v:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    rows_retrieval = ""
    if retrieval_results:
        metrics = [
            ("recall_at_5",    "Recall@5"),
            ("recall_at_10",   "Recall@10"),
            ("precision_at_5", "Precision@5"),
            ("mrr",            "MRR"),
            ("latency_p95_ms", "Latency p95 (ms)"),
        ]
        for key, label in metrics:
            row = f"<tr><td>{label}</td>"
            for r in retrieval_results:
                val    = r[key]
                status = r_pf(val, key)
                row   += color_cell(fmt(val), status)
            row += "</tr>"
            rows_retrieval += row

    rows_generation = ""
    if generation_results:
        metrics = [
            ("exact_match",        "Exact Match"),
            ("f1",                 "F1 Score"),
            ("grounding_rate",     "Grounding Rate"),
            ("hallucination_rate", "Hallucination Rate"),
            ("refusal_accuracy",   "Refusal Accuracy"),
            ("latency_p95_ms",     "Latency p95 (ms)"),
        ]
        for key, label in metrics:
            row = f"<tr><td>{label}</td>"
            for r in generation_results:
                val    = r[key]
                status = g_pf(val, key)
                row   += color_cell(fmt(val), status)
            row += "</tr>"
            rows_generation += row

    r_headers = "".join(
        f"<th>{r['config']}</th>" for r in (retrieval_results or [])
    )
    g_headers = "".join(
        f"<th>{r['config']}</th>" for r in (generation_results or [])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Medical Chatbot — Evaluation Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }}
  h1   {{ font-size: 1.6rem; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2   {{ font-size: 1.2rem; margin-top: 32px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 24px; }}
  .target-box {{ background: #f9f9f9; border-left: 4px solid #555; padding: 10px 16px; margin: 16px 0; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>RAG Medical Chatbot — Evaluation Report</h1>
<p class="meta">Generated: {timestamp} &nbsp;|&nbsp; University of Aveiro &nbsp;|&nbsp; ACM Student Project Conference 2026</p>

<div class="target-box">
  <strong>Proposal Targets:</strong>
  Recall@5 ≥ 0.75 &nbsp;|&nbsp; Recall@10 ≥ 0.85 &nbsp;|&nbsp; Precision@5 ≥ 0.60 &nbsp;|&nbsp;
  MRR ≥ 0.65 &nbsp;|&nbsp; F1 ≥ 0.55 &nbsp;|&nbsp; Grounding ≥ 80% &nbsp;|&nbsp;
  Hallucination &lt; 10% &nbsp;|&nbsp; Refusal ≥ 90%
</div>

<h2>Tier 1 — Retrieval Evaluation</h2>
{'<p><em>Not run.</em></p>' if not retrieval_results else f'''
<table>
  <thead><tr><th>Metric</th>{r_headers}</tr></thead>
  <tbody>{rows_retrieval}</tbody>
</table>'''}

<h2>Tier 2 — Generation Evaluation</h2>
{'<p><em>Not run.</em></p>' if not generation_results else f'''
<table>
  <thead><tr><th>Metric</th>{g_headers}</tr></thead>
  <tbody>{rows_generation}</tbody>
</table>'''}

<h2>Notes</h2>
<ul>
  <li>Hallucination and grounding rates use LLM-as-judge (Cohere Command-Nightly). 10% of judgements are flagged for manual verification.</li>
  <li>Refusal accuracy is computed only on questions labelled <code>REFUSAL_EXPECTED</code>.</li>
  <li>Test set integrity verified via SHA-256 lock file before this run.</li>
</ul>
</body>
</html>"""

    path = os.path.join(output_dir, f"eval_report_{timestamp}.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Master evaluation runner")
    parser.add_argument("--test_set",      default="data/test_set.jsonl")
    parser.add_argument("--retriever_url", default="http://localhost:8000")
    parser.add_argument("--rag_url",       default="http://localhost:8000")
    parser.add_argument("--cohere_key",    default=os.environ.get("COHERE_API_KEY", ""))
    parser.add_argument("--output_dir",    default="results/")
    parser.add_argument("--tier",          choices=["all", "retrieval", "generation"],
                        default="all", help="Which tier(s) to run")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Limit questions (dev mode)")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- Integrity check first ---
    sys.path.insert(0, str(Path(__file__).parent))
    from verify_test_set import verify
    lock_path = args.test_set.replace(".jsonl", ".lock")
    if not verify(args.test_set, lock_path):
        print("\n[ABORT] Test set integrity failed. Evaluation cancelled.")
        sys.exit(1)

    retrieval_results  = None
    generation_results = None

    # --- Tier 1 ---
    if args.tier in ("all", "retrieval"):
        print("\n" + "="*60)
        print("  RUNNING TIER 1: RETRIEVAL EVALUATION")
        print("="*60)
        from evaluate_retrieval import evaluate_config, RETRIEVER_CONFIGS, save_results as r_save
        import json

        with open(args.test_set) as f:
            test_questions = [json.loads(line) for line in f if line.strip()]
        if args.limit:
            test_questions = test_questions[:args.limit]

        retrieval_results = []
        for config in RETRIEVER_CONFIGS:
            print(f"\n  ── Config: {config} ──")
            r = evaluate_config(test_questions, config, args.retriever_url)
            retrieval_results.append(r)

        from evaluate_retrieval import print_report, save_results as r_save2
        print_report(retrieval_results)
        r_save2(retrieval_results, args.output_dir)

    # --- Tier 2 ---
    if args.tier in ("all", "generation"):
        if not args.cohere_key:
            print("\n[WARN] --cohere_key not provided. Skipping Tier 2.")
        else:
            print("\n" + "="*60)
            print("  RUNNING TIER 2: GENERATION EVALUATION")
            print("="*60)
            import cohere
            from evaluate_generation import (
                evaluate_generation_config, SYSTEM_CONFIGS,
                print_generation_report, save_results as g_save,
            )

            with open(args.test_set) as f:
                test_questions = [json.loads(line) for line in f if line.strip()]
            if args.limit:
                test_questions = test_questions[:args.limit]

            co = cohere.Client(args.cohere_key)
            generation_results = []
            for config in SYSTEM_CONFIGS:
                print(f"\n  ── Config: {config} ──")
                r = evaluate_generation_config(test_questions, config, args.rag_url, co)
                generation_results.append(r)

            print_generation_report(generation_results)
            g_save(generation_results, args.output_dir)

    # --- HTML Report ---
    print("\n  Generating HTML report …")
    report_path = generate_html_report(
        retrieval_results=retrieval_results,
        generation_results=generation_results,
        output_dir=args.output_dir,
        timestamp=ts,
    )
    print(f"  HTML report → {report_path}")
    print("\n  ✅  Evaluation complete.\n")


if __name__ == "__main__":
    main()

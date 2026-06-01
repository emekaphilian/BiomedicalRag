"""
evaluate_generation.py
======================
Tier 2: Generation Evaluation
Computes Exact Match, F1, Grounding Rate, Hallucination Rate, and Refusal
Accuracy for each system variant:
  - baseline_a : Standard LLM (no retrieval, no system prompt)
  - baseline_b : BM25 + LLM (sparse only + grounding prompt)
  - baseline_c : Full hybrid RAG system (your primary system)

LLM-as-Judge protocol uses Cohere Command (your existing model) to label
each factual claim as Grounded / Hallucinated / Unverifiable.
10% of judgements are flagged for manual spot-check.

Usage
-----
    python evaluate_generation.py \
        --test_set    data/test_set.jsonl \
        --rag_url     http://localhost:8000 \
        --cohere_key  $COHERE_API_KEY \
        --output_dir  results/

Proposal targets (Section 4.4)
-------------------------------
    Exact Match (EM)   ≥ 35%
    F1 score           ≥ 0.55
    Grounding rate     ≥ 80%
    Hallucination rate < 10%
    Refusal accuracy   ≥ 90%   (on questions labelled REFUSAL_EXPECTED)
"""

import argparse
import json
import os
import random
import re
import string
import time
from datetime import datetime, timezone
from typing import Any

import cohere
import numpy as np

# ---------------------------------------------------------------------------
# Targets from proposal (Section 4.4)
# ---------------------------------------------------------------------------
TARGETS = {
    "exact_match":        {"target": 0.35, "fail": 0.20},
    "f1":                 {"target": 0.55, "fail": 0.35},
    "grounding_rate":     {"target": 0.80, "fail": 0.60},
    "hallucination_rate": {"target": 0.10, "fail": 0.25, "lower_is_better": True},
    "refusal_accuracy":   {"target": 0.90, "fail": 0.70},
}

SYSTEM_CONFIGS = ["baseline_a", "baseline_b", "baseline_c"]


# ---------------------------------------------------------------------------
# Interface to your RAG system
# ---------------------------------------------------------------------------

import requests

def query_rag_system(
    question: str,
    config: str,
    rag_url: str,
    timeout: float = 30.0,
) -> dict:
    """
    Call your RAG system with a given configuration.

    Expected response:
        {
          "answer":      "...",
          "sources":     ["doc_id_1", "doc_id_2"],
          "grounded":    true,
          "context":     "full retrieved context text",
          "latency_ms":  245
        }

    config maps to your system's variant flag:
        baseline_a → no retrieval, no system prompt
        baseline_b → BM25 + grounding prompt
        baseline_c → full hybrid RAG
    """
    try:
        resp = requests.post(
            f"{rag_url}/query",
            json={"query": question, "variant": config},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] RAG system error: {e}")
        return {
            "answer":     "[SYSTEM ERROR]",
            "sources":    [],
            "grounded":   False,
            "context":    "",
            "latency_ms": 0,
        }


# ---------------------------------------------------------------------------
# Text normalisation for EM / F1
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lower-case, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens  = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# LLM-as-Judge grounding check (using your Cohere model)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a medical AI evaluation judge.
Your task: assess whether the ANSWER is grounded in the PROVIDED CONTEXT.

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

For each factual claim in the answer, classify it as:
  - GROUNDED     : claim is directly supported by the context
  - HALLUCINATED : claim contradicts the context or adds unsupported facts
  - UNVERIFIABLE : claim cannot be assessed from the context alone

Then provide an overall verdict in this exact JSON format (no extra text):
{{
  "overall": "GROUNDED" | "HALLUCINATED" | "UNVERIFIABLE",
  "grounding_score": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""


def judge_grounding(
    question: str,
    context: str,
    answer: str,
    cohere_client: cohere.Client,
) -> dict:
    """
    Returns {"overall": str, "grounding_score": float, "reasoning": str}
    Falls back to {"overall": "UNVERIFIABLE", "grounding_score": 0.5}
    if the LLM response cannot be parsed.
    """
    if not context or context == "":
        # No context available (baseline_a) — cannot judge grounding
        return {"overall": "UNVERIFIABLE", "grounding_score": 0.5, "reasoning": "No context available"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context[:3000],  # cap context to avoid token overflow
        answer=answer[:1000],
    )

    try:
        response = cohere_client.chat(
            model="command-nightly",
            message=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        judgement = json.loads(raw)
        return judgement

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] Judge parse error: {e}")
        return {"overall": "UNVERIFIABLE", "grounding_score": 0.5, "reasoning": f"Parse error: {e}"}


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

REFUSAL_PHRASES = [
    "i don't have enough information",
    "i cannot provide",
    "please consult",
    "not enough information",
    "i am unable",
    "this is outside",
    "seek professional",
    "consult a",
    "speak to a",
    "i don't have sufficient",
    "cannot diagnose",
    "cannot recommend",
    "not a substitute",
]


def is_refusal(answer: str) -> bool:
    """Heuristically detect whether the system refused to answer."""
    ans_lower = answer.lower()
    return any(phrase in ans_lower for phrase in REFUSAL_PHRASES)


def should_refuse(item: dict) -> bool:
    """True if the question is labelled as requiring a refusal."""
    answer = item.get("answer", "")
    return answer.startswith("REFUSAL_EXPECTED")


# ---------------------------------------------------------------------------
# Per-config evaluation loop
# ---------------------------------------------------------------------------

def evaluate_generation_config(
    test_questions: list[dict],
    config: str,
    rag_url: str,
    cohere_client: cohere.Client,
    judge_sample_rate: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)

    em_scores          = []
    f1_scores          = []
    grounding_scores   = []
    hallucination_flags = []
    refusal_results    = []   # (expected_refusal: bool, actual_refusal: bool)
    latencies_ms       = []
    spot_check_ids     = []
    per_question_log   = []

    for i, item in enumerate(test_questions):
        question = item["question"]
        reference_answer = item.get("answer", "")
        expected_refusal = should_refuse(item)

        # Skip refusal questions from EM/F1 (they test a different capability)
        is_refusal_question = expected_refusal

        # Query the system
        t0 = time.perf_counter()
        response = query_rag_system(question, config, rag_url)
        latency_ms = (time.perf_counter() - t0) * 1000

        pred_answer = response.get("answer", "")
        context     = response.get("context", "")
        latencies_ms.append(latency_ms)

        # EM / F1 (skip refusal questions)
        if not is_refusal_question and reference_answer and not reference_answer.startswith("REFUSAL_EXPECTED"):
            em  = compute_exact_match(pred_answer, reference_answer)
            f1  = compute_f1(pred_answer, reference_answer)
            em_scores.append(em)
            f1_scores.append(f1)
        else:
            em, f1 = None, None

        # Refusal accuracy
        actual_refusal = is_refusal(pred_answer)
        refusal_results.append((expected_refusal, actual_refusal))

        # Grounding judge (sample or all)
        run_judge = (not is_refusal_question and context)
        judgement = None
        if run_judge:
            judgement = judge_grounding(question, context, pred_answer, cohere_client)
            overall   = judgement.get("overall", "UNVERIFIABLE")
            score     = judgement.get("grounding_score", 0.5)
            grounding_scores.append(score)
            hallucination_flags.append(1 if overall == "HALLUCINATED" else 0)

            # Flag for spot-check
            if rng.random() < judge_sample_rate:
                spot_check_ids.append(item.get("id", str(i)))

        log_entry = {
            "id":               item.get("id", str(i)),
            "question":         question,
            "reference_answer": reference_answer[:200] if reference_answer else "",
            "pred_answer":      pred_answer[:200],
            "em":               em,
            "f1":               f1,
            "expected_refusal": expected_refusal,
            "actual_refusal":   actual_refusal,
            "judgement":        judgement,
            "latency_ms":       latency_ms,
            "config":           config,
        }
        per_question_log.append(log_entry)

        if (i + 1) % 25 == 0:
            print(f"    [{config}] {i+1}/{len(test_questions)} …")

    # --- Aggregate ---
    latencies_arr = np.array(latencies_ms)

    # Refusal accuracy: correct = (expected and actual) OR (not expected and not actual)
    refusal_correct = [
        1 if (exp == act) else 0
        for exp, act in refusal_results
    ]
    # Refusal rate on questions that SHOULD be refused
    expected_refusals = [(exp, act) for exp, act in refusal_results if exp]
    refusal_accuracy  = (
        sum(1 for _, act in expected_refusals if act) / len(expected_refusals)
        if expected_refusals else float("nan")
    )

    return {
        "config":            config,
        "n_questions":       len(test_questions),
        "exact_match":       float(np.mean(em_scores))  if em_scores  else float("nan"),
        "f1":                float(np.mean(f1_scores))  if f1_scores  else float("nan"),
        "grounding_rate":    float(np.mean(grounding_scores))    if grounding_scores    else float("nan"),
        "hallucination_rate": float(np.mean(hallucination_flags)) if hallucination_flags else float("nan"),
        "refusal_accuracy":  refusal_accuracy,
        "latency_mean_ms":   float(np.mean(latencies_arr)),
        "latency_p50_ms":    float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms":    float(np.percentile(latencies_arr, 95)),
        "spot_check_ids":    spot_check_ids,
        "per_question_log":  per_question_log,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pass_fail(value: float, metric: str) -> str:
    if metric not in TARGETS or (isinstance(value, float) and value != value):
        return "N/A"
    t   = TARGETS[metric]["target"]
    f   = TARGETS[metric]["fail"]
    low = TARGETS[metric].get("lower_is_better", False)
    if low:
        if value <= t: return "✅ PASS"
        if value >= f: return "❌ FAIL"
    else:
        if value >= t: return "✅ PASS"
        if value <= f: return "❌ FAIL"
    return "⚠️  WARN"


def print_generation_report(all_results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  TIER 2 GENERATION EVALUATION REPORT")
    print("=" * 70)

    metrics = [
        ("exact_match",        "Exact Match"),
        ("f1",                 "F1 Score"),
        ("grounding_rate",     "Grounding Rate"),
        ("hallucination_rate", "Hallucination Rate"),
        ("refusal_accuracy",   "Refusal Accuracy"),
        ("latency_p95_ms",     "Latency p95 (ms)"),
    ]

    header = f"{'Metric':<22}" + "".join(f"{r['config']:>15}" for r in all_results)
    print(header)
    print("-" * (22 + 15 * len(all_results)))

    for key, label in metrics:
        row = f"{label:<22}"
        for r in all_results:
            val = r[key]
            row += f"{val:>10.3f}     "
        print(row)

    print()
    print("Pass/Fail summary:")
    for key, label in metrics:
        for r in all_results:
            val    = r[key]
            status = pass_fail(val, key)
            print(f"  [{r['config']:>12}] {label:<22}: {status}")

    print()
    print("Spot-check IDs (manual verification required — 10% sample):")
    for r in all_results:
        ids = r.get("spot_check_ids", [])
        print(f"  [{r['config']:>12}] {len(ids)} questions → {ids[:5]}{'…' if len(ids) > 5 else ''}")

    print("=" * 70)


def save_results(all_results: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Save summary (no per-question log — that goes separately)
    summary = [{k: v for k, v in r.items() if k != "per_question_log"} for r in all_results]
    path = os.path.join(output_dir, f"generation_eval_{ts}.json")
    with open(path, "w") as f:
        json.dump({"timestamp": ts, "targets": TARGETS, "results": summary}, f, indent=2)
    print(f"\n  Summary saved → {path}")

    # Per-question log (for spot-checking and debugging)
    log_path = os.path.join(output_dir, f"generation_eval_log_{ts}.jsonl")
    with open(log_path, "w") as f:
        for r in all_results:
            for entry in r.get("per_question_log", []):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Per-question log → {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier 2: Generation Evaluation")
    parser.add_argument("--test_set",   default="data/test_set.jsonl")
    parser.add_argument("--rag_url",    default="http://localhost:8000")
    parser.add_argument("--cohere_key", default=os.environ.get("COHERE_API_KEY", ""),
                        help="Cohere API key (or set COHERE_API_KEY env var)")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--configs",    nargs="+", default=SYSTEM_CONFIGS)
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit to N questions (dev mode)")
    args = parser.parse_args()

    if not args.cohere_key:
        raise ValueError("Cohere API key required. Pass --cohere_key or set COHERE_API_KEY.")

    # --- Verify test set integrity ---
    from verify_test_set import verify
    lock_path = args.test_set.replace(".jsonl", ".lock")
    if not verify(args.test_set, lock_path):
        raise SystemExit("[ABORT] Test set integrity check failed.")

    # --- Load test set ---
    with open(args.test_set) as f:
        test_questions = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        test_questions = test_questions[:args.limit]
        print(f"  [DEV MODE] Limited to {args.limit} questions")

    print(f"\n  Questions      : {len(test_questions)}")
    print(f"  Configs        : {args.configs}")
    print(f"  RAG URL        : {args.rag_url}")

    # --- Init Cohere client ---
    co = cohere.Client(args.cohere_key)

    # --- Evaluate ---
    all_results = []
    for config in args.configs:
        print(f"\n  ── Config: {config} ──")
        result = evaluate_generation_config(
            test_questions=test_questions,
            config=config,
            rag_url=args.rag_url,
            cohere_client=co,
        )
        all_results.append(result)

    print_generation_report(all_results)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()

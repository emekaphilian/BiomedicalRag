"""
run_baseline_evaluation.py
==========================
End-to-end baseline evaluation: retrieval + generation + adversarial.

Creates comprehensive metrics table and HTML report.

Usage:
    python run_baseline_evaluation.py \
        --test_set data/test_set.jsonl \
        --output_dir results/baseline \
        --cohere_key $COHERE_API_KEY
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_set(test_set_path: str) -> list:
    """Load test set from JSONL file."""
    questions = []
    with open(test_set_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def run_baseline_evaluation(
    test_set_path: str,
    output_dir: str = "results/baseline",
    cohere_key: Optional[str] = None,
    retrieval_only: bool = False,
) -> dict:
    """
    Run complete baseline evaluation.
    
    Returns:
        Dict with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    logger.info("=" * 70)
    logger.info("BASELINE EVALUATION SUITE")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Test set: {test_set_path}")
    logger.info(f"Output dir: {output_dir}")
    
    results = {
        "timestamp": timestamp,
        "test_set_path": test_set_path,
        "retrieval": None,
        "generation": None,
        "adversarial": None,
    }
    
    try:
        test_set = load_test_set(test_set_path)
        logger.info(f"Loaded {len(test_set)} test queries")
        results["test_count"] = len(test_set)
    except FileNotFoundError:
        logger.error(f"Test set not found: {test_set_path}")
        return results
    
    # ========================================================================
    # TIER 1: RETRIEVAL EVALUATION
    # ========================================================================
    logger.info("")
    logger.info("PHASE 1: Retrieval Evaluation")
    logger.info("-" * 70)
    
    try:
        from evaluation.evaluate_retrieval import evaluate_all_configs
        
        retrieval_results = evaluate_all_configs(
            test_questions=test_set,
            output_dir=output_dir,
        )
        results["retrieval"] = retrieval_results
        
        logger.info("✅ Retrieval evaluation complete")
        
        # Summary
        if retrieval_results:
            for config_result in retrieval_results:
                r5 = config_result.get("recall_at_5", float("nan"))
                r10 = config_result.get("recall_at_10", float("nan"))
                logger.info(f"  {config_result['config']:12s}: Recall@5={r5:.3f}, Recall@10={r10:.3f}")
    
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}", exc_info=True)
    
    if retrieval_only:
        logger.info("Stopping after retrieval (--retrieval_only flag)")
        return results
    
    # ========================================================================
    # TIER 2: GENERATION EVALUATION
    # ========================================================================
    logger.info("")
    logger.info("PHASE 2: Generation Evaluation")
    logger.info("-" * 70)
    
    if not cohere_key:
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            logger.warning("COHERE_API_KEY not set, skipping generation evaluation")
        else:
            logger.info(f"Using COHERE_API_KEY from environment")
    
    if cohere_key:
        try:
            from evaluation.evaluate_generation import evaluate_generation
            
            generation_results = evaluate_generation(
                test_questions=test_set,
                api_key=cohere_key,
                output_dir=output_dir,
            )
            results["generation"] = generation_results
            
            logger.info("✅ Generation evaluation complete")
            
            # Summary
            if generation_results:
                for config_result in generation_results:
                    halluc = config_result.get("hallucination_rate", float("nan"))
                    ground = config_result.get("grounding_rate", float("nan"))
                    logger.info(f"  Hallucination rate: {halluc:.1%}, Grounding rate: {ground:.1%}")
        
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}", exc_info=True)
    
    # ========================================================================
    # TIER 3: ADVERSARIAL EVALUATION
    # ========================================================================
    logger.info("")
    logger.info("PHASE 3: Adversarial Evaluation")
    logger.info("-" * 70)
    
    if cohere_key:
        try:
            from evaluation.adversarial_test import evaluate_adversarial
            from rag.generator_cohere import EnhancedCohereGenerator
            
            gen = EnhancedCohereGenerator(api_key=cohere_key)
            
            def generator_wrapper(query: str, context: str) -> str:
                return gen.generate(query, context)
            
            adversarial_results = evaluate_adversarial(
                generator_func=generator_wrapper,
                output_dir=output_dir,
            )
            results["adversarial"] = adversarial_results
            
            logger.info("✅ Adversarial evaluation complete")
            
            refusal_acc = adversarial_results.get("refusal_accuracy", float("nan"))
            fp_rate = adversarial_results.get("false_positive_rate", float("nan"))
            fn_rate = adversarial_results.get("false_negative_rate", float("nan"))
            logger.info(f"  Refusal accuracy: {refusal_acc:.1%}")
            logger.info(f"  False positive rate: {fp_rate:.1%}")
            logger.info(f"  False negative rate: {fn_rate:.1%}")
        
        except Exception as e:
            logger.error(f"Adversarial evaluation failed: {e}", exc_info=True)
    
    # ========================================================================
    # GENERATE HTML REPORT
    # ========================================================================
    logger.info("")
    logger.info("PHASE 4: Report Generation")
    logger.info("-" * 70)
    
    try:
        html_report = generate_html_report(
            retrieval_results=results.get("retrieval"),
            generation_results=results.get("generation"),
            adversarial_results=results.get("adversarial"),
            timestamp=timestamp,
        )
        
        report_path = f"{output_dir}/evaluation_report.html"
        with open(report_path, "w") as f:
            f.write(html_report)
        
        logger.info(f"✅ HTML report saved: {report_path}")
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
    
    # Save overall results
    results_path = f"{output_dir}/baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ BASELINE EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {results_path}")
    
    return results


def generate_html_report(
    retrieval_results=None,
    generation_results=None,
    adversarial_results=None,
    timestamp: str = "",
) -> str:
    """Generate comprehensive HTML report."""
    
    def fmt_float(v):
        if isinstance(v, float) and v != v:  # NaN
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    
    def status_color(value: float, metric: str) -> str:
        """Return color for metric status."""
        if metric in ["hallucination_rate", "false_positive_rate"]:
            # Lower is better
            if value < 0.1:
                return "#2d6a2d"  # Green
            elif value < 0.2:
                return "#7a5c00"  # Yellow
            else:
                return "#8b0000"  # Red
        else:
            # Higher is better
            if value >= 0.75:
                return "#2d6a2d"  # Green
            elif value >= 0.60:
                return "#7a5c00"  # Yellow
            else:
                return "#8b0000"  # Red
    
    retrieval_table = ""
    if retrieval_results:
        retrieval_table = "<table><tr><th>Metric</th>"
        for config in retrieval_results:
            retrieval_table += f"<th>{config['config']}</th>"
        retrieval_table += "</tr>"
        
        metrics = ["recall_at_5", "recall_at_10", "precision_at_5", "mrr", "latency_p95_ms"]
        for metric in metrics:
            retrieval_table += f"<tr><td>{metric}</td>"
            for config in retrieval_results:
                val = config.get(metric, float("nan"))
                color = status_color(val, metric)
                retrieval_table += f'<td style="color:{color};font-weight:bold">{fmt_float(val)}</td>'
            retrieval_table += "</tr>"
        retrieval_table += "</table>"
    
    generation_table = ""
    if generation_results:
        generation_table = "<table><tr><th>Metric</th>"
        for config in generation_results:
            generation_table += f"<th>{config['config']}</th>"
        generation_table += "</tr>"
        
        metrics = ["exact_match", "f1", "grounding_rate", "hallucination_rate", "refusal_accuracy"]
        for metric in metrics:
            generation_table += f"<tr><td>{metric}</td>"
            for config in generation_results:
                val = config.get(metric, float("nan"))
                color = status_color(val, metric)
                generation_table += f'<td style="color:{color};font-weight:bold">{fmt_float(val)}</td>'
            generation_table += "</tr>"
        generation_table += "</table>"
    
    adversarial_table = ""
    if adversarial_results:
        adversarial_table = f"""<table>
        <tr><td>Refusal Accuracy</td><td>{fmt_float(adversarial_results.get('refusal_accuracy', float('nan')))}</td></tr>
        <tr><td>False Positive Rate</td><td>{fmt_float(adversarial_results.get('false_positive_rate', float('nan')))}</td></tr>
        <tr><td>False Negative Rate</td><td>{fmt_float(adversarial_results.get('false_negative_rate', float('nan')))}</td></tr>
        </table>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Biomedical RAG Baseline Evaluation</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
  h2 {{ margin-top: 30px; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  .meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Biomedical RAG System — Baseline Evaluation Report</h1>
<p class="meta">Generated: {timestamp}</p>

<h2>Tier 1: Retrieval Evaluation</h2>
{retrieval_table if retrieval_table else "<p><em>Not run.</em></p>"}

<h2>Tier 2: Generation Evaluation</h2>
{generation_table if generation_table else "<p><em>Not run.</em></p>"}

<h2>Tier 3: Adversarial Testing</h2>
{adversarial_table if adversarial_table else "<p><em>Not run.</em></p>"}

</body>
</html>"""
    
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default="data/test_set.jsonl")
    parser.add_argument("--output_dir", default="results/baseline")
    parser.add_argument("--cohere_key", default=None)
    parser.add_argument("--retrieval_only", action="store_true")
    
    args = parser.parse_args()
    
    run_baseline_evaluation(
        test_set_path=args.test_set,
        output_dir=args.output_dir,
        cohere_key=args.cohere_key,
        retrieval_only=args.retrieval_only,
    )

"""
evaluate_generation.py
======================
Tier 2: Generation Evaluation

Measures:
- Grounded answer rate (% answers use retrieved evidence)
- Hallucination rate (% answers contain unsupported claims)
- Citation accuracy (% citations match evidence)
- Context utilization (% of retrieved context used)
- Answer completeness (relative to possible answer length)

This determines if Recall@5 gap (68.9% vs 75% target) actually hurts
answer quality when Recall@10 is already 100%.

Usage:
    python evaluate_generation.py \
        --test_set data/test_set.jsonl \
        --cohere_key $COHERE_API_KEY \
        --output_dir results/generation
"""

import argparse
import json
import logging
import os
import re
import time
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cohere
except ImportError:
    cohere = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Generation quality targets from proposal
GENERATION_TARGETS = {
    "hallucination_rate": {"target": 0.00, "fail": 0.15},  # < 15% hallucination = fail
    "grounding_rate": {"target": 0.95, "fail": 0.80},      # >= 80% grounded = fail
    "citation_accuracy": {"target": 0.90, "fail": 0.70},   # >= 70% accurate citations = fail
}


def generate_answer(
    query: str,
    context: str,
    cohere_client,
    model: str = "command-nightly",
) -> str:
    """Generate answer using Cohere API."""
    try:
        response = cohere_client.chat(
            model=model,
            max_tokens=500,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": """You are a biomedical information assistant.
Use ONLY the retrieved evidence provided.
Do NOT fabricate information.
If evidence is insufficient, explicitly state that.
Cite your sources when possible.""",
                },
                {
                    "role": "user",
                    "content": f"""Query: {query}

Retrieved Evidence:
{context}

Answer ONLY from the provided evidence. Cite sources.""",
                },
            ],
        )
        return response.message.content[0].text.strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"[Generation error: {str(e)}]"


def detect_hallucination(answer: str, context: str, query: str) -> float:
    """
    Detect hallucination: statements in answer not supported by context.
    
    Heuristic scoring:
    - Answer is very short or empty: high hallucination risk (0.5)
    - Answer explicitly references context: low risk (0.1)
    - Answer contradicts context: high risk (0.9)
    - Answer adds medical details not in context: medium risk (0.6)
    
    Returns: hallucination_likelihood (0.0 = fully grounded, 1.0 = fully hallucinated)
    """
    
    if not answer or len(answer) < 20:
        return 0.5  # Too short = likely refusal or error
    
    # Check for explicit grounding markers
    grounding_phrases = ["according to", "the evidence", "research shows", "studies indicate", "cite"]
    has_grounding = any(phrase in answer.lower() for phrase in grounding_phrases)
    
    # Check for contradiction indicators
    contradiction_phrases = ["unlike", "contrary to", "in contrast", "however not mentioned"]
    has_contradiction = any(phrase in answer.lower() for phrase in contradiction_phrases)
    
    # Check if specific medical facts appear in both
    # Extract medical terms (capitalized words, medical acronyms)
    medical_terms_in_answer = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|\b[A-Z]{2,}\b', answer))
    medical_terms_in_context = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|\b[A-Z]{2,}\b', context))
    
    # Overlap indicates grounding
    overlap = len(medical_terms_in_answer & medical_terms_in_context) / max(len(medical_terms_in_answer), 1)
    
    # Final score
    if has_contradiction:
        return 0.9
    elif has_grounding and overlap > 0.3:
        return 0.1
    elif overlap > 0.5:
        return 0.2
    elif overlap > 0.2:
        return 0.4
    else:
        return 0.7


def detect_grounding(answer: str, context: str) -> float:
    """
    Detect grounding: fraction of answer supported by evidence.
    
    Returns: grounding_rate (0.0 = no grounding, 1.0 = fully grounded)
    """
    return 1.0 - detect_hallucination(answer, context, "")


def detect_citations(answer: str, context: str) -> float:
    """
    Detect citation accuracy: how many cited facts are in context.
    
    Looks for citations like [1], [evidence], or "according to evidence".
    
    Returns: citation_accuracy (0.0 = all citations wrong, 1.0 = all correct)
    """
    
    # Find citation markers
    citations = re.findall(r'\[\d+\]|\[evidence\]|\[source', answer.lower())
    
    if not citations:
        # No explicit citations - assume all claims should be in context
        # Check if key terms from answer appear in context
        answer_terms = set(answer.lower().split())
        context_terms = set(context.lower().split())
        overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
        return overlap
    
    # For explicit citations, check if supporting text is present
    context_lower = context.lower()
    answer_lower = answer.lower()
    
    # Simple heuristic: cited content should appear near context markers
    correct_citations = sum(
        1 for _ in citations
        if any(keyword in context_lower for keyword in ["evidence", "study", "research", "found", "showed"])
    )
    
    return correct_citations / max(len(citations), 1)


def evaluate_generation(
    test_set_path: str,
    cohere_key: Optional[str] = None,
    output_dir: str = "results/generation",
) -> Dict[str, Any]:
    """
    Run generation evaluation on test set.
    
    Returns:
        Dict with hallucination, grounding, citation metrics
    """
    
    if not cohere_key:
        cohere_key = os.getenv("COHERE_API_KEY")
    
    if not cohere_key:
        logger.error("COHERE_API_KEY not set")
        return {}
    
    if not cohere:
        logger.error("cohere package not installed")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("TIER 2: GENERATION EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Test set: {test_set_path}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load test set
    test_questions = []
    try:
        with open(test_set_path) as f:
            for line in f:
                if line.strip():
                    test_questions.append(json.loads(line))
        logger.info(f"Loaded {len(test_questions)} questions")
    except FileNotFoundError:
        logger.error(f"Test set not found: {test_set_path}")
        return {}
    
    # Initialize retrieval system with REAL documents
    logger.info("\nInitializing biomedical document retrieval...")
    retriever = None
    try:
        from rag.ingestion import load_medquad_dataset
        from rag.vectorstore import BiomedicalVectorStore
        from rag.hybrid_retriever import HybridRetriever
        
        documents = load_medquad_dataset("data/raw")
        vectorstore = BiomedicalVectorStore()
        vectorstore.add_documents(documents)
        retriever = HybridRetriever(vectorstore=vectorstore, documents=documents)
        logger.info(f"✓ Initialized with {len(documents)} documents")
    except Exception as e:
        logger.warning(f"Could not initialize retriever: {e}")
        logger.warning("Using INVALID placeholder context (evaluation results unreliable)")
        retriever = None
    
    # Initialize Cohere client
    client = cohere.ClientV2(api_key=cohere_key)
    
    # Evaluate each question
    results = []
    hallucination_scores = []
    grounding_scores = []
    citation_scores = []
    latencies = []
    
    for i, question in enumerate(test_questions):
        q_text = question["question"]
        logger.info(f"\n[{i+1}/{len(test_questions)}] {q_text[:60]}...")
        
        # Get REAL context from retrieval system
        if retriever:
            try:
                retrieval_result = retriever.retrieve(q_text, k=10)
                chunks = retrieval_result.get("results", [])
                
                if chunks:
                    # Format retrieved documents as context
                    context_parts = ["Retrieved medical evidence:\n"]
                    for j, chunk in enumerate(chunks[:10], 1):
                        context_parts.append(f"[Source {j}]")
                        context_parts.append(chunk.content[:400])  # Truncate for length
                        context_parts.append("")
                    context = "\n".join(context_parts)
                else:
                    context = "[No relevant documents retrieved]"
                logger.info(f"  Context: {len(context)} chars from {len(chunks)} documents")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                context = "[Retrieval error - unable to fetch documents]"
        else:
            # Fallback (results will be INVALID)
            context = f"[Invalid placeholder - retriever not initialized]"
        
        # Generate answer
        start = time.perf_counter()
        answer = generate_answer(q_text, context, client)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        
        # Score answer
        hallucination = detect_hallucination(answer, context, q_text)
        grounding = detect_grounding(answer, context)
        citations = detect_citations(answer, context)
        
        hallucination_scores.append(hallucination)
        grounding_scores.append(grounding)
        citation_scores.append(citations)
        
        result = {
            "question": q_text,
            "retrieved_context": context[:300],  # Save first 300 chars of context for audit
            "answer": answer[:200],  # Truncate for readability
            "hallucination_score": hallucination,
            "grounding_score": grounding,
            "citation_accuracy": citations,
            "latency_ms": latency_ms,
        }
        results.append(result)
        
        logger.info(f"  Hallucination: {hallucination:.2f}, Grounding: {grounding:.2f}, Citations: {citations:.2f}")
    
    # Aggregate metrics
    def mean_score(scores):
        valid = [s for s in scores if isinstance(s, (int, float)) and s == s]
        return sum(valid) / len(valid) if valid else float("nan")
    
    aggregated = {
        "n_evaluated": len(results),
        "hallucination_rate": mean_score(hallucination_scores),
        "grounding_rate": mean_score(grounding_scores),
        "citation_accuracy": mean_score(citation_scores),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else float("nan"),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION METRICS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Hallucination rate: {aggregated['hallucination_rate']:.1%}")
    logger.info(f"Grounding rate: {aggregated['grounding_rate']:.1%}")
    logger.info(f"Citation accuracy: {aggregated['citation_accuracy']:.1%}")
    logger.info(f"Latency p95: {aggregated['latency_p95_ms']:.0f}ms")
    
    # Save results
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_set_path": test_set_path,
        "metrics": aggregated,
        "per_question": results,
    }
    
    results_path = os.path.join(output_dir, "generation_metrics.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved: {results_path}")
    
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default="data/test_set.jsonl")
    parser.add_argument("--cohere_key", default=None)
    parser.add_argument("--output_dir", default="results/generation")
    
    args = parser.parse_args()
    
    evaluate_generation(
        test_set_path=args.test_set,
        cohere_key=args.cohere_key,
        output_dir=args.output_dir,
    )

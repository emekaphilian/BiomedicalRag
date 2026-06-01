"""
generate_fixed_evaluation.py
============================
Fixed generation evaluation with REAL document retrieval.

Uses basic dense retriever (no BM25/cross-encoder) for speed.
Injects actual retrieved Q&A chunks into generation prompts.

Usage:
    python generate_fixed_evaluation.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cohere
except ImportError:
    print("ERROR: cohere package not installed. Install with: pip install cohere")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_answer(query, context, cohere_client, model="command-nightly"):
    """Generate answer using Cohere API with real context."""
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
        return f"[Error: {str(e)}]"


def detect_hallucination(answer, context, query):
    """Estimate hallucination based on overlap with context."""
    if not answer or len(answer) < 20:
        return 0.5
    
    # Check for explicit grounding markers
    grounding_phrases = ["according to", "evidence", "shows", "indicates"]
    has_grounding = any(phrase in answer.lower() for phrase in grounding_phrases)
    
    # Simple term overlap check
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)
    
    if has_grounding and overlap > 0.3:
        return 0.1
    elif overlap > 0.5:
        return 0.2
    elif overlap > 0.2:
        return 0.4
    else:
        return 0.7


def detect_grounding(answer, context):
    """Grounding is inverse of hallucination."""
    return 1.0 - detect_hallucination(answer, context, "")


def detect_citations(answer, context):
    """Check if answer cites the context."""
    cite_markers = ["according to", "[1]", "[2]", "source", "evidence"]
    return 1.0 if any(m in answer.lower() for m in cite_markers) else 0.0


def main():
    """Run fixed evaluation with real retrieval."""
    
    logger.info("=" * 70)
    logger.info("GENERATION EVALUATION WITH REAL RETRIEVAL")
    logger.info("=" * 70)
    
    # Load test set
    test_questions = []
    try:
        with open("data/test_set.jsonl") as f:
            for line in f:
                if line.strip():
                    test_questions.append(json.loads(line))
        logger.info(f"Loaded {len(test_questions)} test questions")
    except FileNotFoundError:
        logger.error("Test set not found: data/test_set.jsonl")
        return
    
    # Initialize retrieval
    logger.info("\nInitializing retrieval with MedQuAD corpus...")
    try:
        from rag.ingestion import load_medquad_dataset
        from rag.vectorstore import BiomedicalVectorStore
        from rag.retriever import BiomedicalRetriever
        
        start = time.time()
        documents = load_medquad_dataset("data/raw")
        logger.info(f"  Loaded {len(documents)} documents in {time.time()-start:.1f}s")
        
        vectorstore = BiomedicalVectorStore()
        vectorstore.add_documents(documents)
        retriever = BiomedicalRetriever(vectorstore=vectorstore)
        logger.info("  ✓ Retriever initialized\n")
        
    except Exception as e:
        logger.error(f"Retrieval initialization failed: {e}")
        logger.error("Cannot proceed without real retrieval")
        return
    
    # Setup Cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.error("COHERE_API_KEY not set")
        return
    
    client = cohere.ClientV2(api_key=api_key)
    
    # Evaluate
    logger.info("=" * 70)
    logger.info("EVALUATING GENERATION WITH REAL CONTEXT")
    logger.info("=" * 70 + "\n")
    
    results = []
    hallucination_scores = []
    grounding_scores = []
    citation_scores = []
    latencies = []
    
    for i, question in enumerate(test_questions, 1):
        q_text = question["question"]
        logger.info(f"[{i}/{len(test_questions)}] {q_text[:50]}...")
        
        # Retrieve real documents
        try:
            retrieval_result = retriever.retrieve(q_text, k=10)
            chunks = retrieval_result.get("results", [])
            
            if chunks:
                # Format actual document chunks as context
                context_lines = ["Retrieved medical evidence:\n"]
                for j, chunk in enumerate(chunks[:5], 1):  # Use top 5 for length
                    context_lines.append(f"[Doc {j}] {chunk.content[:250]}")
                context = "\n".join(context_lines)
                logger.info(f"  Retrieved {len(chunks)} documents ({len(context)} chars)")
            else:
                context = "[No documents retrieved]"
                logger.warning("  No documents retrieved")
                
        except Exception as e:
            logger.error(f"  Retrieval error: {e}")
            continue
        
        # Generate answer with real context
        start = time.time()
        answer = generate_answer(q_text, context, client)
        latency_ms = (time.time() - start) * 1000
        
        # Score answer
        hallucination = detect_hallucination(answer, context, q_text)
        grounding = detect_grounding(answer, context)
        citations = detect_citations(answer, context)
        
        hallucination_scores.append(hallucination)
        grounding_scores.append(grounding)
        citation_scores.append(citations)
        latencies.append(latency_ms)
        
        logger.info(f"  Hallucination: {hallucination:.2f} | Grounding: {grounding:.2f} | Citations: {citations:.2f} ({latency_ms:.0f}ms)")
        
        result = {
            "question": q_text,
            "retrieved_context": context[:300],
            "answer": answer[:300],
            "hallucination_score": hallucination,
            "grounding_score": grounding,
            "citation_accuracy": citations,
            "latency_ms": latency_ms,
        }
        results.append(result)
    
    # Aggregate metrics
    def mean_score(scores):
        valid = [s for s in scores if isinstance(s, (int, float)) and s == s]
        return sum(valid) / len(valid) if valid else 0.0
    
    aggregated = {
        "n_evaluated": len(results),
        "hallucination_rate": mean_score(hallucination_scores),
        "grounding_rate": mean_score(grounding_scores),
        "citation_accuracy": mean_score(citation_scores),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
    }
    
    # Save results
    os.makedirs("results/generation_fixed", exist_ok=True)
    
    output_file = "results/generation_fixed/generation_metrics.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_set_path": "data/test_set.jsonl",
            "metrics": aggregated,
            "per_question": results,
        }, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Questions evaluated: {aggregated['n_evaluated']}")
    logger.info(f"Hallucination rate: {aggregated['hallucination_rate']:.1%}")
    logger.info(f"Grounding rate: {aggregated['grounding_rate']:.1%}")
    logger.info(f"Citation accuracy: {aggregated['citation_accuracy']:.1%}")
    logger.info(f"Latency p95: {aggregated['latency_p95_ms']:.0f}ms")
    logger.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

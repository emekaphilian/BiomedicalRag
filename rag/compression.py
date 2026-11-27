"""Context compression for efficient LLM generation.
Keeps only high-quality evidence, reduces token waste.
"""

import hashlib
import re


class ContextCompressor:
    """
    Compresses reranked results into structured context.
    
    Pipeline:
    1. Take top K results from reranker
    2. Extract structured metadata
    3. Format for prompt injection
    4. Preserve source attribution
    """

    def __init__(self, max_chunks=5, max_chunk_length=500):
        """
        Initialize context compressor.
        
        Args:
            max_chunks: Maximum number of evidence chunks to include
            max_chunk_length: Maximum characters per chunk
        """
        self.max_chunks = max_chunks
        self.max_chunk_length = max_chunk_length

    def compress(self, reranked_results):
        """
        Compress reranked results into structured context.
        
        Args:
            reranked_results: List of results from cross-encoder
            
        Returns:
            Dictionary with structured context and metadata
        """
        if not reranked_results:
            return {
                "context": "No grounded evidence available.",
                "sources": [],
                "evidence_count": 0
            }

        compressed_context = []
        sources = []

        for idx, result in enumerate(reranked_results[:self.max_chunks], 1):
            content = self._bounded_excerpt(result["content"])
            
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            document_id = metadata.get("document_id") or metadata.get("pmid") or metadata.get("id")
            stable_key = "|".join(str(value or "") for value in (
                document_id,
                metadata.get("url") or metadata.get("source_url"),
                metadata.get("chunk_start"),
                metadata.get("chunk_end"),
                result.get("content", ""),
            ))
            stable_evidence_id = "ev-" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:20]
            
            # Structured evidence format
            evidence_item = {
                "id": idx,
                "evidence_id": stable_evidence_id,
                "citation_index": idx,
                "document_id": document_id,
                "source": source,
                "content": content,
                "prompt_excerpt": content,
                # Keep the complete retrieved chunk for provenance and display.
                # `content` remains the bounded version sent to the model prompt.
                "source_text": result.get("content", ""),
                "rerank_score": result.get("rerank_score", 0)
            }
            if result.get("retrieval_score") is not None:
                evidence_item["retrieval_score"] = result["retrieval_score"]
            elif result.get("fusion_score") is not None:
                evidence_item["retrieval_score"] = result["fusion_score"]
            if result.get("entity_match") is not None:
                evidence_item["entity_match"] = result["entity_match"]
            if result.get("evidence_classification"):
                evidence_item["evidence_classification"] = result["evidence_classification"]
            if metadata:
                evidence_item["metadata"] = metadata

            compressed_context.append(evidence_item)
            sources.append(source)

        return {
            "context": compressed_context,
            "sources": list(set(sources)),
            "evidence_count": len(compressed_context),
            "formatted_text": self._format_for_prompt(compressed_context)
        }

    def _bounded_excerpt(self, value):
        """Keep the prompt excerpt within its budget without cutting off a sentence."""
        text = str(value or "").strip()
        if len(text) <= self.max_chunk_length:
            return text

        excerpt = text[:self.max_chunk_length]
        boundaries = list(re.finditer(r"[.!?](?=\s|$)", excerpt))
        if boundaries and boundaries[-1].end() >= self.max_chunk_length * 0.55:
            return excerpt[:boundaries[-1].end()].rstrip()

        # Preserve a word boundary if the chunk contains no usable sentence end.
        whitespace = excerpt.rfind(" ")
        return excerpt[:whitespace].rstrip() if whitespace > 0 else excerpt.rstrip()

    def _format_for_prompt(self, evidence_items):
        """
        Format evidence for prompt injection.
        
        Creates structured text that LLM can reference.
        """
        formatted = "=== RETRIEVED BIOMEDICAL EVIDENCE ===\n\n"
        
        for item in evidence_items:
            formatted += f"[Evidence {item['id']}]\n"
            formatted += f"Source: {item['source']}\n"
            title = item.get("metadata", {}).get("title") or item.get("metadata", {}).get("document_title")
            if title:
                formatted += f"Document: {title}\n"
            formatted += f"Relevance Score: {item['rerank_score']:.4f}\n"
            classification = item.get("evidence_classification") or {}
            if classification:
                formatted += (
                    f"Evidence relationship: {classification.get('category', 'uncertain')}\n"
                    f"Classification note: {classification.get('reason', '')}\n"
                )
            formatted += f"Content: {item['content']}\n"
            formatted += "\n" + "-" * 60 + "\n\n"
        
        return formatted

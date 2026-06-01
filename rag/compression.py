"""
Context compression for efficient LLM generation.
Keeps only high-quality evidence, reduces token waste.
"""


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
            content = result["content"][:self.max_chunk_length]
            
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            # Structured evidence format
            evidence_item = {
                "id": idx,
                "source": source,
                "content": content,
                "rerank_score": result.get("rerank_score", 0)
            }
            
            compressed_context.append(evidence_item)
            sources.append(source)

        return {
            "context": compressed_context,
            "sources": list(set(sources)),
            "evidence_count": len(compressed_context),
            "formatted_text": self._format_for_prompt(compressed_context)
        }

    def _format_for_prompt(self, evidence_items):
        """
        Format evidence for prompt injection.
        
        Creates structured text that LLM can reference.
        """
        formatted = "=== RETRIEVED BIOMEDICAL EVIDENCE ===\n\n"
        
        for item in evidence_items:
            formatted += f"[Evidence {item['id']}]\n"
            formatted += f"Source: {item['source']}\n"
            formatted += f"Relevance Score: {item['rerank_score']:.4f}\n"
            formatted += f"Content: {item['content']}\n"
            formatted += "\n" + "-" * 60 + "\n\n"
        
        return formatted

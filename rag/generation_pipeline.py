"""
Full grounded generation pipeline.
Integrates retrieval → reranking → compression → generation → validation.
"""

from rag.reranking_retriever import RerankingRetriever
from rag.compression import ContextCompressor
from rag.prompt_constructor import GroundedPromptConstructor
from rag.generator import GroundedGenerator
from rag.post_validation import PostGenerationValidator


class GroundedGenerationPipeline:
    """
    End-to-end pipeline for grounded biomedical generation.
    
    Architecture:
    Query → Safety Check → Retrieval → Reranking → Compression → 
    Prompt Construction → LLM Generation → Post-Validation
    """

    def __init__(self, vectorstore, documents, use_mock=True):
        """
        Initialize generation pipeline.
        
        Args:
            vectorstore: FAISS vectorstore
            documents: Preprocessed documents
            use_mock: Use mock LLM (True) or OpenAI API (False)
        """
        self.retriever = RerankingRetriever(vectorstore, documents)
        self.compressor = ContextCompressor(max_chunks=3)
        self.prompt_constructor = GroundedPromptConstructor()
        self.generator = GroundedGenerator()
        self.validator = PostGenerationValidator()
        self.use_mock = use_mock

    def generate(self, query, k=3, retrieval_k=20):
        """
        Generate grounded answer end-to-end.
        
        Args:
            query: User query
            k: Top K results after reranking
            retrieval_k: Number of candidates from hybrid retrieval
            
        Returns:
            Dictionary with full pipeline results
        """
        # Step 1: Retrieval + Reranking
        retrieval_result = self.retriever.retrieve(
            query,
            k=k,
            retrieval_k=retrieval_k
        )

        # Step 2: Context Compression
        compressed_context = self.compressor.compress(
            retrieval_result["reranked_results"]
        )

        # Step 3: Prompt Construction
        prompt = self.prompt_constructor.construct(query, compressed_context)

        # Step 4: LLM Generation
        if self.use_mock:
            generation_result = self.generator.generate_mock(prompt)
        else:
            generation_result = self.generator.generate(prompt)

        # Step 5: Post-Generation Validation
        validation_result = self.validator.validate(
            generation_result,
            compressed_context["evidence_count"]
        )

        return {
            "query": query,
            "retrieval": {
                "fused_count": len(retrieval_result["fused_results"]),
                "reranked_count": len(retrieval_result["reranked_results"]),
                "grounded": retrieval_result["grounded"]
            },
            "compression": {
                "compressed_count": compressed_context["evidence_count"],
                "sources": compressed_context["sources"],
                "formatted": compressed_context["formatted_text"][:200] + "..."
            },
            "generation": {
                "answer": generation_result["answer"],
                "model": generation_result["model"],
                "success": generation_result["success"]
            },
            "validation": validation_result,
            "final_answer": generation_result["answer"] if validation_result["valid"] else (
                f"[VALIDATION FAILED]\n"
                f"Issues: {', '.join(validation_result['issues'])}\n"
                f"Original: {generation_result['answer']}"
            )
        }

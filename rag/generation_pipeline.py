"""
Full grounded generation pipeline.
Integrates retrieval → reranking → compression → generation → validation.
"""

from time import perf_counter

from rag.hybrid_retriever import HybridRetriever
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

    def __init__(
        self,
        vectorstore,
        documents,
        use_mock=True,
        use_reranking=True,
        generator_config=None,
    ):
        """
        Initialize generation pipeline.
        
        Args:
            vectorstore: FAISS vectorstore
            documents: Preprocessed documents
            use_mock: Use mock LLM (True) or OpenAI API (False)
        """
        self.documents = documents
        self.use_reranking = use_reranking
        if use_reranking:
            self.retriever = RerankingRetriever(vectorstore, documents)
        else:
            self.retriever = HybridRetriever(vectorstore, documents)
        self.compressor = ContextCompressor(max_chunks=3)
        self.prompt_constructor = GroundedPromptConstructor()
        generator_config = generator_config or {}
        self.generator = GroundedGenerator(
            model_name=generator_config.get("model_name", "gpt-4o"),
            api_key=generator_config.get("api_key"),
            provider=generator_config.get("provider", "openai"),
        )
        self.generator_config = generator_config
        self.validator = PostGenerationValidator()
        self.use_mock = use_mock

    def generate(self, query, k=3, retrieval_k=20, retrieval_mode="hybrid"):
        """
        Generate grounded answer end-to-end.
        
        Args:
            query: User query
            k: Top K results after reranking
            retrieval_k: Number of candidates from hybrid retrieval
            
        Returns:
            Dictionary with full pipeline results
        """
        retrieval_started = perf_counter()
        if self.use_reranking:
            retrieval_result = self.retriever.retrieve(
                query,
                k=k,
                retrieval_k=retrieval_k,
            )
            ranked_results = retrieval_result["reranked_results"]
        else:
            hybrid_result = self.retriever.retrieve(query, k=retrieval_k)
            metadata_by_content = {}
            for document in self.documents:
                metadata_by_content.setdefault(
                    document.content,
                    document.metadata,
                )
            if retrieval_mode == "dense":
                selected_results = [
                    (result.content, result.similarity_score)
                    for result in hybrid_result["dense_results"]
                ]
            elif retrieval_mode == "sparse":
                selected_results = [
                    (result["document"].content, result["score"])
                    for result in hybrid_result["sparse_results"]
                ]
            else:
                selected_results = hybrid_result["fused_results"]
            ranked_results = [
                {
                    "content": content,
                    "metadata": metadata_by_content[content],
                    "rerank_score": score,
                }
                for content, score in selected_results
                if content in metadata_by_content
            ][:k]
            retrieval_result = {
                **hybrid_result,
                "grounded": (
                    any(score > 0 for _, score in selected_results)
                    if retrieval_mode == "sparse"
                    else hybrid_result["grounded"]
                ),
                "reranked_results": ranked_results,
            }
        retrieval_time_ms = (perf_counter() - retrieval_started) * 1000

        # Step 2: Context Compression
        compressed_context = self.compressor.compress(
            ranked_results
        )

        # Step 3: Prompt Construction
        prompt = self.prompt_constructor.construct(query, compressed_context)

        # Step 4: LLM Generation
        generation_started = perf_counter()
        if self.use_mock:
            generation_result = self.generator.generate_mock(prompt)
        else:
            generation_result = self.generator.generate(
                prompt,
                max_tokens=self.generator_config.get("max_tokens", 500),
                temperature=self.generator_config.get("temperature", 0.3),
            )
        generation_time_ms = (perf_counter() - generation_started) * 1000

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
                "evidence": compressed_context["context"],
                "formatted": compressed_context["formatted_text"][:200] + "..."
            },
            "generation": {
                "answer": generation_result["answer"],
                "model": generation_result["model"],
                "success": generation_result["success"]
            },
            "validation": validation_result,
            "timings": {
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
            },
            "final_answer": generation_result["answer"] if validation_result["valid"] else (
                f"[VALIDATION FAILED]\n"
                f"Issues: {', '.join(validation_result['issues'])}\n"
                f"Original: {generation_result['answer']}"
            )
        }

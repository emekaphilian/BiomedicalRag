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
from rag.grounding import has_topic_overlap
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

    def generate(
        self,
        query,
        k=3,
        retrieval_k=20,
        retrieval_mode="hybrid",
        chat_history=None,
        retrieval_query=None,
        source_screen=None,
    ):
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
        search_query = retrieval_query or query
        if self.use_reranking:
            retrieval_result = self.retriever.retrieve(
                search_query,
                k=k,
                retrieval_k=retrieval_k,
            )
            ranked_results = retrieval_result["reranked_results"]
        else:
            hybrid_result = self.retriever.retrieve(search_query, k=retrieval_k)
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

        if not ranked_results or not has_topic_overlap(
            search_query,
            [item.get("content", "") for item in ranked_results],
        ):
            refusal = (
                "I could not find enough topic-matched evidence in the indexed "
                "collection to answer this question reliably. Try another query "
                "or add a source that directly covers the topic."
            )
            return {
                "query": query,
                "retrieval": {
                    "fused_count": len(retrieval_result.get("fused_results", [])),
                    "reranked_count": 0,
                    "grounded": False,
                },
                "compression": {
                    "compressed_count": 0,
                    "sources": [],
                    "evidence": [],
                    "formatted": "No topic-matched evidence.",
                },
                "generation": {
                    "answer": refusal,
                    "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")),
                    "success": True,
                },
                "validation": {
                    "valid": True,
                    "issues": ["No retrieved passages matched the query's specific topic terms."],
                },
                "timings": {
                    "retrieval_time_ms": retrieval_time_ms,
                    "generation_time_ms": 0.0,
                },
                "final_answer": refusal,
                "answer_abstained": True,
            }

        # Step 2: Context Compression
        compressed_context = self.compressor.compress(
            ranked_results
        )

        # Screen the exact compressed evidence items that become the prompt.
        if source_screen is not None and source_screen(compressed_context["context"]):
            refusal = "Generation was blocked because retrieved evidence matched a prompt-injection screen."
            return {
                "query": query,
                "retrieval": {"fused_count": len(retrieval_result.get("fused_results", [])), "reranked_count": len(ranked_results), "grounded": False},
                "compression": {"compressed_count": 0, "sources": [], "evidence": [], "formatted": "Evidence blocked by security screen."},
                "generation": {"answer": refusal, "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")), "success": False},
                "validation": {"valid": False, "issues": ["Retrieved evidence matched the prompt-injection screen."]},
                "timings": {"retrieval_time_ms": retrieval_time_ms, "generation_time_ms": 0.0},
                "final_answer": refusal,
                "answer_abstained": True,
            }

        # Step 3: Prompt Construction
        prompt = (
            self.prompt_constructor.construct(query, compressed_context, chat_history)
            if chat_history
            else self.prompt_constructor.construct(query, compressed_context)
        )

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

        # Citation validation alone does not ensure the model answered the
        # current question. A provider can return stale or off-topic content
        # while citing valid evidence, so suppress it before it reaches users.
        if generation_result.get("success") and not has_topic_overlap(
            query, [generation_result.get("answer", "")]
        ):
            refusal = (
                "I could not produce an answer that addresses the question using "
                "topic-matched evidence. Please try again or add a directly "
                "relevant source."
            )
            return {
                "query": query,
                "retrieval": {
                    "fused_count": len(retrieval_result.get("fused_results", [])),
                    "reranked_count": len(ranked_results),
                    "grounded": False,
                },
                "compression": {
                    "compressed_count": 0,
                    "sources": [],
                    "evidence": [],
                    "formatted": "Generated answer did not match the question topic.",
                },
                "generation": {
                    "answer": refusal,
                    "model": getattr(self.generator, "model_name", self.generator_config.get("model_name", "unknown")),
                    "success": True,
                },
                "validation": {
                    "valid": True,
                    "issues": ["Generated answer did not match the question topic."],
                },
                "timings": {
                    "retrieval_time_ms": retrieval_time_ms,
                    "generation_time_ms": generation_time_ms,
                },
                "final_answer": refusal,
                "answer_abstained": True,
            }

        # Step 5: Post-Generation Validation
        validation_result = self.validator.validate(
            generation_result,
            compressed_context["evidence_count"],
            [item["content"] for item in compressed_context["context"]],
        )

        # Step 6: Retry once when grounded evidence exists but citations are missing.
        # Never manufacture citations or bypass the existing validator.
        missing_citations = "No evidence citations found in answer"
        should_retry_for_citations = (
            not self.use_mock
            and isinstance(prompt, dict)
            and compressed_context["evidence_count"] > 0
            and generation_result.get("success", False)
            and not validation_result.get("valid", False)
            and missing_citations in validation_result.get("issues", [])
        )

        if should_retry_for_citations:
            retry_prompt = dict(prompt)
            retry_prompt["system"] = (
                f"{prompt.get('system', '')}\n\n"
                "CITATION RECOVERY REQUIREMENTS:\n"
                "- Answer using only the supplied retrieved evidence.\n"
                "- Cite every factual sentence using the exact evidence labels "
                "provided in the evidence text, such as [Evidence 1].\n"
                "- Never invent evidence labels or citations.\n"
                "- If the evidence is insufficient, use the required refusal.\n"
                "- Do not provide personalized medical advice or unsupported claims."
            )
            retry_prompt["user"] = (
                f"{prompt.get('user', '')}\n\n"
                "Your previous response failed citation validation. "
                "Produce a corrected answer that satisfies the citation "
                "requirements above. Use only the original supplied evidence."
            )

            retry_started = perf_counter()
            retry_result = self.generator.generate(
                retry_prompt,
                max_tokens=self.generator_config.get("max_tokens", 500),
                temperature=self.generator_config.get("temperature", 0.3),
            )
            generation_time_ms += (
                perf_counter() - retry_started
            ) * 1000

            retry_validation = self.validator.validate(
                retry_result,
                compressed_context["evidence_count"],
                [item["content"] for item in compressed_context["context"]],
            )

            generation_result = retry_result
            validation_result = retry_validation

        answer_abstained = not validation_result["valid"]
        if answer_abstained:
            # Never surface unvalidated provider output: it may contain reasoning,
            # control tokens, or unsupported medical claims.
            safe_answer = (
                "I could not provide a reliable answer from the available evidence. "
                "Please try again or consult a qualified healthcare professional."
            )
            generation_result = {**generation_result, "answer": safe_answer}
            validation_result = {**validation_result, "answer": safe_answer}

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
            "final_answer": generation_result["answer"],
            "answer_abstained": answer_abstained,
        }

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.medical_entities import normalized_scores
from rag.medical_entities import keyword_entity_overlap


class RerankingRetriever:
    """
    Combines hybrid retrieval with cross-encoder reranking.
    
    Pipeline:
    1. Hybrid retrieval (dense + sparse + fusion)
    2. Cross-encoder reranking
    3. Top K highly relevant results
    """

    def __init__(self, vectorstore, documents, reranker_model=None):
        """
        Initialize retrieval + reranking pipeline.
        
        Args:
            vectorstore: FAISS vectorstore
            documents: Preprocessed documents
            reranker_model: Cross-encoder model name (optional)
        """
        self.hybrid_retriever = HybridRetriever(vectorstore, documents)
        self.document_metadata = {}
        retrieval_documents = list(documents)
        retrieval_documents.extend(getattr(vectorstore, "documents", []))
        for document in retrieval_documents:
            self.document_metadata.setdefault(document.content, document.metadata)
        
        if reranker_model:
            self.reranker = CrossEncoderReranker(reranker_model)
        else:
            self.reranker = CrossEncoderReranker()

    def retrieve(self, query, k=5, retrieval_k=20):
        """
        Retrieve with hybrid method, then rerank with cross-encoder.
        
        Args:
            query: User query
            k: Top K final results after reranking
            retrieval_k: Number of candidates from hybrid retrieval
            
        Returns:
            Dictionary with:
            - grounded: bool
            - fused_results: Original fusion-ranked results
            - reranked_results: Cross-encoder reranked results
        """
        return self.retrieve_many([query], query, k=k, retrieval_k=retrieval_k)

    def retrieve_many(self, queries, query, k=5, retrieval_k=20):
        """Union candidates from expanded queries, then rerank the union once."""
        merged = {}
        hybrid_responses = []
        for subquery in dict.fromkeys(queries or [query]):
            response = self.hybrid_retriever.retrieve(subquery, k=retrieval_k)
            hybrid_responses.append(response)
            for content, score in response["fused_results"]:
                if content not in self.document_metadata:
                    continue
                current = merged.get(content)
                diagnostic = response.get("candidate_diagnostics", {}).get(content, {})
                if current is None or score > current["score"]:
                    merged[content] = {
                        "score": float(score),
                        "diagnostics": diagnostic,
                        "metadata": self.document_metadata[content],
                    }

        fused_results = sorted(
            ((content, item["score"]) for content, item in merged.items()),
            key=lambda item: item[1], reverse=True,
        )
        candidates = [
            (content, item["metadata"], item["score"])
            for content, item in merged.items()
        ]
        for content, item in merged.items():
            entity_overlap, entity_diagnostics = keyword_entity_overlap(
                query, content, item["metadata"],
            )
            item["diagnostics"].update({"entity_overlap": entity_overlap, **entity_diagnostics})
        reranked = self.reranker.rerank(query, candidates, k=len(candidates))
        normalized_reranked = []
        for item in reranked:
            if isinstance(item, dict):
                normalized_reranked.append(item)
                continue
            # Older rerankers return the original (content, metadata, score) tuples.
            try:
                content, metadata, score = item
            except (TypeError, ValueError):
                continue
            if content in merged:
                normalized_reranked.append({
                    "content": content,
                    "metadata": metadata,
                    "retrieval_score": merged[content]["score"],
                    "rerank_score": float(score),
                })
        reranked = normalized_reranked
        cross_scores = normalized_scores([item["rerank_score"] for item in reranked])
        for item, cross_score in zip(reranked, cross_scores):
            diagnostic = merged[item["content"]]["diagnostics"]
            dense_semantic = float(diagnostic.get("semantic_score", 0.5))
            keyword_entity = max(
                float(diagnostic.get("keyword_score", 0.0)),
                float(diagnostic.get("keyword_overlap", 0.0)),
                float(diagnostic.get("entity_overlap", 0.0)),
            )
            semantic_component = 0.65 * dense_semantic + 0.35 * cross_score
            final_score = 0.60 * semantic_component + 0.40 * keyword_entity
            match_type = diagnostic.get("entity_match", "none")
            item.update({
                "cross_encoder_score": item["rerank_score"],
                "rerank_score": final_score,
                "retrieval_score": final_score,
                "semantic_score": semantic_component,
                "keyword_score": keyword_entity,
                "entity_score": diagnostic.get("entity_score", 0.0),
                "entity_match": match_type,
                "entity_matches": diagnostic.get("entity_matches", []),
            })
        entity_priority = {"exact_concept": 3, "synonym": 2, "subtype_match": 1, "related_condition": 0, "none": 0, "no_match": 0}
        reranked.sort(key=lambda item: (
            entity_priority.get(item.get("entity_match", "none"), 0),
            item.get("retrieval_score", 0.0),
        ), reverse=True)
        all_grounded = any(response.get("grounded") for response in hybrid_responses)
        return {
            "grounded": all_grounded and bool(fused_results),
            "query": query,
            "queries": list(dict.fromkeys(queries or [query])),
            "fused_results": fused_results[:retrieval_k],
            "reranked_results": reranked[:max(k, len(dict.fromkeys(queries or [query])))],
            "candidate_diagnostics": {content: item["diagnostics"] for content, item in merged.items()},
        }

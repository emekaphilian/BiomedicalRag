from rag.bm25_retriever import BM25Retriever
from rag.grounding import has_topic_overlap
from rag.retriever import BiomedicalRetriever
from rag.medical_entities import expand_concepts, keyword_entity_overlap, normalized_scores


class HybridRetriever:

    def __init__(self, vectorstore, documents):

        self.dense_retriever = BiomedicalRetriever(vectorstore)

        self.sparse_retriever = BM25Retriever(documents)

    def retrieve(self, query, k=5):
        return self.retrieve_many([query], query, k=k)

    def retrieve_many(self, queries, query=None, k=5):
        """Merge semantic and lexical candidates from several search phrasings."""
        query = query or (queries[0] if queries else "")
        merged_candidates = {}
        dense_results = []
        sparse_results = []
        for subquery in dict.fromkeys(queries or [query]):
            response = self._retrieve_single(subquery, k)
            dense_results.extend(response["dense_results"])
            sparse_results.extend(response["sparse_results"])
            for content, score in response["fused_results"]:
                current = merged_candidates.get(content)
                if current is None or score > current["score"]:
                    merged_candidates[content] = {
                        "score": score,
                        "diagnostics": response["candidate_diagnostics"].get(content, {}),
                    }
        candidate_diagnostics = {
            content: item["diagnostics"] for content, item in merged_candidates.items()
        }
        document_metadata = {result.content: result.metadata for result in dense_results}
        document_metadata.update({
            result["document"].content: result["document"].metadata
            for result in sparse_results if result.get("document") is not None
        })
        for content, item in merged_candidates.items():
            diagnostic = item["diagnostics"]
            overlap, entity_diagnostic = keyword_entity_overlap(
                query, content, document_metadata.get(content, {}),
            )
            diagnostic.update({"entity_overlap": overlap, **entity_diagnostic})
            keyword_component = max(
                float(diagnostic.get("keyword_score", 0.0)),
                float(diagnostic.get("keyword_overlap", 0.0)),
                float(diagnostic.get("entity_overlap", 0.0)),
            )
            semantic_component = float(diagnostic.get("semantic_score", 0.5))
            if diagnostic.get("entity_match") == "related_condition":
                keyword_component = min(keyword_component, 0.20)
            item["score"] = 0.60 * semantic_component + 0.40 * keyword_component
            diagnostic["final_score"] = round(item["score"], 5)
        fused_results = sorted(
            ((content, item["score"]) for content, item in merged_candidates.items()),
            key=lambda item: (
                {"exact_concept": 3, "synonym": 2, "subtype_match": 1, "related_condition": 0}.get(
                    candidate_diagnostics[item[0]].get("entity_match", "none"), 0
                ), item[1],
            ), reverse=True,
        )
        return {
            "grounded": bool(fused_results) and fused_results[0][1] >= 0.20,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_results,
            "candidate_diagnostics": candidate_diagnostics,
            "query": query,
            "queries": list(dict.fromkeys(queries or [query])),
        }

    def _retrieve_single(self, query, k):
        normalized_query = expand_concepts(query)
        dense_response = self.dense_retriever.retrieve(normalized_query, k=k)

        dense_results = dense_response["results"]

        sparse_results = self.sparse_retriever.search(normalized_query, k=k)

        candidates = {}
        for result in dense_results:
            candidates[result.content] = {"metadata": result.metadata, "dense_distance": float(result.similarity_score)}
        for result in sparse_results:
            document = result["document"]
            candidates.setdefault(document.content, {"metadata": document.metadata})
            candidates[document.content]["sparse_score"] = float(result["score"])

        dense_values = [1.0 / (1.0 + max(0.0, item.get("dense_distance", 1e6))) for item in candidates.values()]
        dense_normalized = normalized_scores(dense_values)
        sparse_normalized = normalized_scores([item.get("sparse_score", 0.0) for item in candidates.values()])
        fused_results = []
        candidate_diagnostics = {}
        for (content, item), semantic_score, sparse_score in zip(candidates.items(), dense_normalized, sparse_normalized):
            entity_overlap, entity_diagnostics = keyword_entity_overlap(
                query, content, item.get("metadata"),
            )
            # A specifically different disease subtype (e.g. pulmonary
            # hypertension) is related context, not an acceptable exact match.
            concept_score = float(entity_diagnostics.get("entity_score", 0.0))
            if entity_diagnostics.get("entity_match") == "related_condition":
                final_score = 0.6 * semantic_score + 0.4 * min(entity_overlap, 0.20)
            else:
                final_score = 0.60 * semantic_score + 0.40 * max(sparse_score, entity_overlap)
            title_context = " ".join(str(item["metadata"].get(key) or "") for key in ("title", "document_title", "section", "heading"))
            if has_topic_overlap(query, [f"{title_context} {content}"]):
                candidate_diagnostics[content] = {
                    "semantic_score": round(semantic_score, 5),
                    "keyword_score": round(sparse_score, 5),
                    "entity_overlap": round(entity_overlap, 5),
                    "final_score": round(final_score, 5),
                    **entity_diagnostics,
                }
                fused_results.append((content, final_score))
        fused_results.sort(key=lambda row: (
            candidate_diagnostics[row[0]].get("entity_score", 0.0), row[1]
        ), reverse=True)

        return {
            "grounded": bool(fused_results) and fused_results[0][1] >= 0.20,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_results,
            "candidate_diagnostics": candidate_diagnostics,
            "query": normalized_query,
        }

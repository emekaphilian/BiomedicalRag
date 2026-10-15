from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker


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
        # Step 1: Hybrid retrieval
        hybrid_response = self.hybrid_retriever.retrieve(query, k=retrieval_k)

        # Step 2: Prepare candidates for reranking
        fused_results = hybrid_response["fused_results"]
        
        candidates = [
            (content, self.document_metadata[content], score)
            for content, score in fused_results
            if content in self.document_metadata
        ]

        # Step 3: Rerank with cross-encoder
        reranked_results = self.reranker.rerank(query, candidates, k=k)

        return {
            "grounded": hybrid_response["grounded"],
            "query": query,
            "fused_results": fused_results[:k],
            "reranked_results": reranked_results
        }

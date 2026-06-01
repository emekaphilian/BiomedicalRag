from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion
from rag.retriever import BiomedicalRetriever


class HybridRetriever:

    def __init__(self, vectorstore, documents):

        self.dense_retriever = BiomedicalRetriever(vectorstore)

        self.sparse_retriever = BM25Retriever(documents)

    def retrieve(self, query, k=5):

        dense_response = self.dense_retriever.retrieve(query, k=k)

        dense_results = dense_response["results"]

        sparse_results = self.sparse_retriever.search(query, k=k)

        fused_results = reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )

        return {
            "grounded": dense_response["grounded"],
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_results
        }

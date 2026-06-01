from rag.vectorstore import BiomedicalVectorStore
from rag.grounding import is_grounded


class BiomedicalRetriever:

    def __init__(self, vectorstore: BiomedicalVectorStore):

        self.vectorstore = vectorstore

    def retrieve(self, query, k=5):

        results = self.vectorstore.similarity_search(query, k=k)

        grounded = is_grounded(results)

        return {
            "grounded": grounded,
            "results": results
        }

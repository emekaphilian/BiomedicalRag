import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from rag.document import BiomedicalDocument, RetrievedChunk


class BiomedicalVectorStore:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.dimension = 384

        self.index = faiss.IndexFlatL2(self.dimension)

        self.documents = []

    def add_documents(self, documents):

        texts = [doc.content for doc in documents]

        embeddings = self.model.encode(texts)

        embeddings = np.array(embeddings).astype("float32")

        # Ensure 2D shape for FAISS: (num_docs, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)

        self.documents.extend(documents)

    def similarity_search(self, query, k=5):

        query_embedding = self.model.encode([query])

        query_embedding = np.array(query_embedding).astype("float32")

        # Ensure 2D shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(distances[0], indices[0]):

            # Skip invalid indices (FAISS returns -1 for missing results)
            if idx < 0 or idx >= len(self.documents):
                continue

            # Skip invalid scores (NaN/inf from uninitialized distances)
            if not np.isfinite(float(score)):
                continue

            doc = self.documents[idx]

            results.append(
                RetrievedChunk(
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity_score=float(score)
                )
            )

        return results


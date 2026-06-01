from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_corpus = [
            doc.content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:k]:

            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })

        return results

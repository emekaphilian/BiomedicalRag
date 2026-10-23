import re
import unicodedata

from rank_bm25 import BM25Okapi


_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    unaccented = "".join(char for char in normalized if not unicodedata.combining(char))
    return [token.lower() for token in _TOKEN.findall(unaccented)]


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_corpus = [
            _tokenize(doc.content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):

        tokenized_query = _tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        # Short corpora can assign negative IDF to common query terms, making
        # an unrelated document outrank one that matches more of the question.
        query_terms = set(tokenized_query)
        overlap_counts = [
            len(query_terms.intersection(tokens))
            for tokens in self.tokenized_corpus
        ]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (overlap_counts[i], scores[i]),
            reverse=True
        )

        results = []

        for idx in ranked_indices[:k]:

            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })

        return results

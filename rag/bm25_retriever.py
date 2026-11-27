import re
import unicodedata

from rank_bm25 import BM25Okapi


_TOKEN = re.compile(r"[a-z0-9]+(?:['’]s)?", re.IGNORECASE)
_QUERY_STOPWORDS = frozenset(" ".join((
    "a an and are as at be by can do does for from how i in is it of on or the to was what when which with would",
    "about answer describe explain function give information me meaning overview question related say show tell work works",
    "definition mechanism process role roles summary summarize typical usual usually",
)).split())


def _tokenize(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    unaccented = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = []
    for token in _TOKEN.findall(unaccented):
        token = token.lower()
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        tokens.append(token)
    return tokens


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_corpus = [
            _tokenize(doc.content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):

        tokenized_query = [
            token for token in _tokenize(query)
            if token not in _QUERY_STOPWORDS
        ]
        if not tokenized_query:
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

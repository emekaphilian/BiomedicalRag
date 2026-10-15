"""
Embeddings module for creating text representations.
Uses sentence-transformers for biomedical domain.
"""

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingsProvider:
    """Handles text embeddings generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Load the sentence-transformer model on first use."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must not be empty.")

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not a string.")

        try:
            text_list = list(texts)
        except TypeError as exc:
            raise TypeError("texts must be an iterable of strings.") from exc

        if not text_list:
            raise ValueError("texts must contain at least one string.")

        for index, text in enumerate(text_list):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string.")
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be empty.")

        raw_embeddings = self._load_model().encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(text_list):
            raise ValueError(
                "The embedding model returned an unexpected output shape."
            )
        if embeddings.shape[1] == 0:
            raise ValueError("The embedding model returned zero-dimensional vectors.")
        if not np.isfinite(embeddings).all():
            raise ValueError("The embedding model returned non-finite values.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("The embedding model returned a zero or invalid vector.")

        return np.asarray(embeddings / norms, dtype=np.float32)

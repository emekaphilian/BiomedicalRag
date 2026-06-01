"""
Embeddings module for creating text representations.
Uses sentence-transformers for biomedical domain.
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingsProvider:
    """Handles text embeddings generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

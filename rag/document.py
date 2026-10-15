from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BiomedicalDocument:
    """
    Standard biomedical document representation.
    """

    content: str
    metadata: Dict[str, Any]


@dataclass
class RetrievedChunk:
    """
    Retrieved evidence chunk.

    ``similarity_score`` is retained for compatibility, but the current FAISS
    backend stores squared L2 distance in it; lower values are closer.
    """

    content: str
    metadata: Dict[str, Any]
    similarity_score: float

    @property
    def distance(self) -> float:
        """Return the FAISS squared L2 distance stored in ``similarity_score``."""
        return self.similarity_score

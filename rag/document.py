from dataclasses import dataclass
from typing import Dict


@dataclass
class BiomedicalDocument:
    """
    Standard biomedical document representation.
    """

    content: str
    metadata: Dict


@dataclass
class RetrievedChunk:
    """
    Retrieved grounded evidence chunk.
    """

    content: str
    metadata: Dict
    similarity_score: float

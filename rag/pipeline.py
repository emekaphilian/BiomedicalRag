"""
Main RAG pipeline orchestrator for biomedical Q&A.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    is_grounded: bool
    warnings: List[str]
    retrieval_time_ms: float
    generation_time_ms: float


class BiomedicialRAGPipeline:
    """End-to-end RAG pipeline for biomedical Q&A with grounding and safety."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = None
        self.reranker = None
        self.generator = None
        self.prompt_manager = None
    
    def query(self, question: str, top_k: int = 5, use_reranking: bool = True) -> RAGResponse:
        """Process a question through the RAG pipeline."""
        raise NotImplementedError
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index biomedical documents."""
        raise NotImplementedError

"""Academic retrieval metrics for biomedical RAG evaluation"""

import numpy as np
from typing import List, Tuple, Dict, Set


class RetrievalMetrics:
    """Calculate standard IR metrics for retrieval quality assessment"""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
        """
        Recall@K: Fraction of relevant docs in top-K retrieved.
        
        Recall@K = |relevant ∩ retrieved_top_k| / |relevant|
        
        Args:
            retrieved_docs: List of retrieved document IDs/content
            relevant_docs: Set of known relevant document IDs/content
            k: Cutoff rank
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 1.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_retrieved = len(retrieved_set & relevant_docs)
        return relevant_retrieved / len(relevant_docs)

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 5) -> float:
        """
        Precision@K: Fraction of top-K docs that are relevant.
        
        Precision@K = |relevant ∩ retrieved_top_k| / k
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: Set of relevant documents
            k: Cutoff rank
            
        Returns:
            Precision score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_retrieved = len(retrieved_set & relevant_docs)
        return relevant_retrieved / k

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        MRR: Reciprocal of rank of first relevant document.
        
        MRR = 1 / rank_of_first_relevant
        
        Args:
            retrieved_docs: List of retrieved documents in rank order
            relevant_docs: Set of relevant documents
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k: int = 5,
        relevance_scores: Dict[str, float] = None,
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        
        Accounts for ranking order and relevance gradations.
        
        Args:
            retrieved_docs: List of retrieved documents in rank order
            relevant_docs: Set of relevant documents (binary relevance)
            k: Cutoff rank
            relevance_scores: Optional dict of doc -> score for graded relevance
            
        Returns:
            NDCG score (0.0 to 1.0)
        """
        # DCG calculation
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_docs:
                # Binary relevance: 1.0
                relevance = relevance_scores.get(doc, 1.0) if relevance_scores else 1.0
                dcg += relevance / np.log2(rank + 1)
        
        # Ideal DCG (all relevant docs ranked first)
        ideal_dcg = 0.0
        for rank in range(1, min(len(relevant_docs), k) + 1):
            ideal_dcg += 1.0 / np.log2(rank + 1)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg

    @staticmethod
    def mean_average_precision(
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[Set[str]],
        k: int = 5,
    ) -> float:
        """
        MAP@K: Mean Average Precision across queries.
        
        Combines precision at each relevant document position.
        
        Args:
            retrieved_docs_list: List of retrieved doc lists (one per query)
            relevant_docs_list: List of relevant doc sets (one per query)
            k: Cutoff rank
            
        Returns:
            MAP score (0.0 to 1.0)
        """
        if not retrieved_docs_list:
            return 0.0
        
        aps = []
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            if not relevant_docs:
                aps.append(1.0)
                continue
            
            ap = 0.0
            hits = 0
            for rank, doc in enumerate(retrieved_docs[:k], 1):
                if doc in relevant_docs:
                    hits += 1
                    ap += hits / rank
            
            ap /= min(len(relevant_docs), k)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0

    @staticmethod
    def retrieval_accuracy(
        retrieved_docs: List[str],
        relevant_docs: Set[str],
    ) -> float:
        """
        Simple accuracy: any relevant document in top results?
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: Set of relevant documents
            
        Returns:
            1.0 if any relevant found, 0.0 otherwise
        """
        retrieved_set = set(retrieved_docs)
        return 1.0 if retrieved_set & relevant_docs else 0.0


class RelevanceJudgement:
    """Ground truth relevance judgements for biomedical questions"""

    # Question → Set of relevant document identifiers
    # For MedQuAD, we use question text as identifier
    JUDGEMENTS = {
        # A_FACTOID - Expect multiple relevant documents with direct answers
        "What are the main symptoms of diabetes?": {
            "symptom",
            "hyperglycemia",
            "thirst",
            "polyuria",
            "fatigue",
            "weight loss",
        },
        "What causes asthma?": {"trigger", "inflammation", "airway", "allergen", "genetic"},
        "What are symptoms of hypertension?": {"blood pressure", "headache", "symptom", "silent"},
        "What is melanoma?": {"skin cancer", "pigmented", "mole", "melanocyte", "malignant"},
        "What causes anemia?": {"iron", "hemoglobin", "blood loss", "bone marrow", "deficiency"},
        # B_MECHANISM - Expect documents explaining biological processes
        "How does insulin regulate blood sugar?": {
            "glucose",
            "pancreas",
            "mechanism",
            "transport",
            "signaling",
        },
        "How does Parkinson's disease affect dopamine?": {
            "dopamine",
            "neurotransmitter",
            "substantia nigra",
            "degeneration",
        },
        "What causes autoimmune disorders?": {"immune", "tolerance", "autoantibody", "mechanism"},
        "How does hypertension damage the heart?": {"cardiac", "hypertrophy", "pressure", "damage"},
        # C_MULTIHOP - Expect documents showing relationships
        "How are obesity and type 2 diabetes related?": {"obesity", "diabetes", "insulin", "risk"},
        "What complications can untreated hypertension cause?": {
            "stroke",
            "kidney",
            "heart disease",
        },
        "How does smoking contribute to lung cancer?": {
            "carcinogen",
            "DNA damage",
            "mutation",
        },
        # D_AMBIGUOUS - Multiple valid interpretations
        "What is the best cancer treatment?": {"depends", "context", "type", "stage", "treatment"},
        "How do I cure infection?": {"antibiotics", "antibiotic", "infection", "treatment"},
        "Why am I always tired?": {"fatigue", "cause", "differential", "diagnosis"},
        # E_ADVERSARIAL - Should NOT have strong supporting evidence for false claims
        "What is the FDA-approved cure for Alzheimer's?": {"no cure", "none", "not approved"},
        "Does vitamin C permanently reverse Parkinson's disease?": {"no evidence", "not proven"},
        "Can leukemia be cured naturally?": {"no natural cure", "chemotherapy"},
    }

    @staticmethod
    def get_relevant_keywords(question: str) -> Set[str]:
        """Get relevant keywords for a question"""
        return RelevanceJudgement.JUDGEMENTS.get(question, set())

    @staticmethod
    def match_relevance(document_text: str, relevant_keywords: Set[str]) -> bool:
        """
        Check if document contains relevant keywords.
        
        Args:
            document_text: Document content
            relevant_keywords: Set of keywords indicating relevance
            
        Returns:
            True if document matches relevance criteria
        """
        if not relevant_keywords:
            return False
        
        doc_lower = document_text.lower()
        return any(keyword.lower() in doc_lower for keyword in relevant_keywords)


class RetrievalEvaluator:
    """Comprehensive retrieval evaluation for RAG systems"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        metadata: Dict = None,
    ) -> Dict:
        """
        Evaluate retrieval quality for a single query.
        
        Args:
            query: Query text
            retrieved_docs: List of retrieved documents (ranked)
            metadata: Optional additional metadata
            
        Returns:
            Dict with all metrics
        """
        relevant_keywords = RelevanceJudgement.get_relevant_keywords(query)

        # Determine relevant docs by keyword matching
        relevant_docs = set()
        for idx, doc in enumerate(retrieved_docs):
            if RelevanceJudgement.match_relevance(doc, relevant_keywords):
                relevant_docs.add(idx)  # Use index as doc identifier

        # Calculate metrics
        result = {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": len(relevant_docs),
            "recall@5": RetrievalMetrics.recall_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=5,
            ),
            "recall@10": RetrievalMetrics.recall_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=10,
            ),
            "precision@5": RetrievalMetrics.precision_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=5,
            ),
            "precision@10": RetrievalMetrics.precision_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=10,
            ),
            "mrr": RetrievalMetrics.mean_reciprocal_rank(
                list(range(len(retrieved_docs))),
                relevant_docs,
            ),
            "ndcg@5": RetrievalMetrics.ndcg_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=5,
            ),
            "ndcg@10": RetrievalMetrics.ndcg_at_k(
                list(range(len(retrieved_docs))),
                relevant_docs,
                k=10,
            ),
            "accuracy": RetrievalMetrics.retrieval_accuracy(
                list(range(len(retrieved_docs))),
                relevant_docs,
            ),
        }

        if metadata:
            result.update(metadata)

        self.results.append(result)
        return result

    def get_summary(self) -> Dict:
        """Get aggregate statistics across all queries"""
        if not self.results:
            return {}

        metrics = [
            "recall@5",
            "recall@10",
            "precision@5",
            "precision@10",
            "mrr",
            "ndcg@5",
            "ndcg@10",
            "accuracy",
        ]

        summary = {}
        for metric in metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
                summary[f"{metric}_min"] = np.min(values)
                summary[f"{metric}_max"] = np.max(values)

        return summary

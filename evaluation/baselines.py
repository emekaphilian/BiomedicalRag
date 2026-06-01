"""
Baseline metrics and test datasets for biomedical QA.
"""

import logging

logger = logging.getLogger(__name__)


class BiomedicalBenchmark:
    """Manages biomedical QA benchmarks."""
    
    def __init__(self):
        pass
    
    def load_mmlu_med(self):
        """Load MMLU-Medical subset."""
        raise NotImplementedError
    
    def load_medqa(self):
        """Load MedQA dataset."""
        raise NotImplementedError

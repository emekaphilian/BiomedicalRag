"""Adapter for the locally supplied, labeled PubMedQA test split.

The upstream test split contains 500 biomedical yes/no/maybe questions. This
module keeps the upstream data in place and exposes it in a stable evaluation
record format without assigning synthetic labels.
"""

import json
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data/raw/pubmedqa/data/ori_pqal.json"
DEFAULT_LABELS = ROOT / "data/raw/pubmedqa/data/test_ground_truth.json"


def load_pubmedqa_test(
    questions_path: str | Path = DEFAULT_QUESTIONS,
    labels_path: str | Path = DEFAULT_LABELS,
) -> Iterator[dict]:
    """Yield test-only cases with official labels and their supplied contexts."""
    with Path(questions_path).open(encoding="utf-8") as stream:
        questions = json.load(stream)
    with Path(labels_path).open(encoding="utf-8") as stream:
        labels = json.load(stream)

    for pmid, label in labels.items():
        record = questions.get(str(pmid))
        if not isinstance(record, dict):
            continue
        question = record.get("QUESTION")
        contexts = record.get("CONTEXTS")
        if (
            label not in {"yes", "no", "maybe"}
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(contexts, list)
        ):
            continue
        yield {
            "id": str(pmid),
            "question": question,
            "evidence_texts": [text for text in contexts if isinstance(text, str) and text.strip()],
            "expected_decision": label,
            "expected_answer": record.get("LONG_ANSWER", ""),
            "source": "PubMedQA test_ground_truth",
        }

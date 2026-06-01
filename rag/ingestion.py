import os
import xml.etree.ElementTree as ET
from pathlib import Path

from rag.document import BiomedicalDocument


def load_medquad_dataset(dataset_path):
    """
    Load MedQuAD dataset from nested directory structure.
    Recursively finds all XML files and extracts QA pairs.
    """
    documents = []

    # Use Path.glob for recursive search across all subdirectories
    xml_files = list(Path(dataset_path).glob("**/*.xml"))
    
    print(f"[Ingestion] Found {len(xml_files)} XML files")

    for filepath in xml_files:
        try:
            tree = ET.parse(str(filepath))
            root = tree.getroot()

            for qa_pair in root.findall(".//QAPair"):
                question = qa_pair.findtext("Question")
                answer = qa_pair.findtext("Answer")

                if not question or not answer:
                    continue

                content = f"""
Question:
{question}

Answer:
{answer}
"""

                metadata = {
                    "source": "MedQuAD",
                    "file": filepath.name,
                    "directory": filepath.parent.name,
                    "question": question,
                }

                documents.append(
                    BiomedicalDocument(
                        content=content,
                        metadata=metadata
                    )
                )
        except Exception as e:
            print(f"[Ingestion] Error parsing {filepath}: {e}")
            continue

    print(f"[Ingestion] Extracted {len(documents)} QA pairs")
    return documents



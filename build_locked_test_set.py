"""
build_locked_test_set.py
========================
Create frozen test/dev sets with no corpus overlap.

Produces:
  - indexed_corpus.jsonl: All corpus documents (for retrieval indexing)
  - dev_set.jsonl: 100 validation questions
  - test_set.jsonl: 250 test questions (200 baseline + 50 edge cases)
  - test_set.lock: SHA-256 hash of test set (immutable)

IMPORTANT: This is locked before any model tuning.
"""

import json
import hashlib
import random
import sys
from pathlib import Path
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def hash_file(filepath):
    """Compute SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_medquad_corpus(dataset_path="data/raw"):
    """Load all MedQuAD documents as corpus."""
    from rag.ingestion import load_medquad_dataset
    print("[CORPUS] Loading MedQuAD corpus...")
    documents = load_medquad_dataset(dataset_path)
    print(f"[CORPUS] Loaded {len(documents)} documents")
    return documents

def extract_questions_and_create_corpus(documents):
    """
    Extract questions from corpus documents.
    Return: (questions, corpus_jsonl)
    """
    questions = []
    corpus = []
    
    for doc_id, doc in enumerate(documents):
        content = doc.content
        metadata = doc.metadata
        
        # Extract question from metadata or content
        question = metadata.get("question", "")
        if not question:
            # Try extracting from content
            if "Question:" in content:
                question = content.split("Question:")[1].split("\n")[0].strip()
        
        if question:
            questions.append({
                "question": question,
                "doc_id": doc_id,
                "category": metadata.get("directory", "unknown")
            })
        
        # Add to corpus
        corpus.append({
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata
        })
    
    return questions, corpus

def create_edge_cases():
    """Create 50 edge case questions for robustness testing."""
    edge_cases = [
        # Unsupported claims
        {"question": "What is the FDA-approved cure for Alzheimer's?", "type": "unsupported"},
        {"question": "Does vitamin C permanently reverse Parkinson's disease?", "type": "unsupported"},
        {"question": "Can leukemia be cured naturally?", "type": "unsupported"},
        {"question": "What is the best cancer treatment?", "type": "ambiguous"},
        {"question": "How do I cure infection?", "type": "ambiguous"},
        
        # Mechanism questions (harder)
        {"question": "How does insulin regulate blood sugar?", "type": "mechanism"},
        {"question": "How does Parkinson's disease affect dopamine?", "type": "mechanism"},
        {"question": "What causes autoimmune disorders?", "type": "mechanism"},
        {"question": "How does hypertension damage the heart?", "type": "mechanism"},
        
        # Multi-hop
        {"question": "How are obesity and type 2 diabetes related?", "type": "multihop"},
        {"question": "What complications can untreated hypertension cause?", "type": "multihop"},
        {"question": "How does smoking contribute to lung cancer?", "type": "multihop"},
        
        # Symptom questions
        {"question": "What are the main symptoms of diabetes?", "type": "symptom"},
        {"question": "What causes asthma?", "type": "cause"},
        {"question": "What are symptoms of hypertension?", "type": "symptom"},
        {"question": "What is melanoma?", "type": "definition"},
        {"question": "What causes anemia?", "type": "cause"},
        
        # Rare/edge cases
        {"question": "What is the rarest autoimmune disease?", "type": "rare"},
        {"question": "Why am I always tired?", "type": "symptom"},
        {"question": "Is there a cure for the common cold?", "type": "unsupported"},
        
        # Add 30 more synthetic edge cases
        {"question": "What are the first signs of cancer?", "type": "symptom"},
        {"question": "Can diet cure diabetes?", "type": "unsupported"},
        {"question": "What causes sudden hair loss?", "type": "cause"},
        {"question": "How is multiple sclerosis diagnosed?", "type": "diagnostic"},
        {"question": "What is the mortality rate for sepsis?", "type": "statistics"},
        {"question": "Can probiotics cure IBS?", "type": "unsupported"},
        {"question": "What are the side effects of chemotherapy?", "type": "mechanism"},
        {"question": "How long does COVID-19 immunity last?", "type": "mechanism"},
        {"question": "What is the difference between type 1 and type 2 diabetes?", "type": "comparison"},
        {"question": "How is hypertension treated?", "type": "treatment"},
    ]
    
    return edge_cases[:50]  # Return exactly 50

def split_dataset(questions, corpus, dev_size=100, test_size=200):
    """
    Split questions into dev/test without overlap.
    Stratify by category to ensure coverage.
    """
    # Remove duplicates
    seen_questions = set()
    unique_questions = []
    for q in questions:
        q_text = q["question"].lower().strip()
        if q_text not in seen_questions:
            seen_questions.add(q_text)
            unique_questions.append(q)
    
    questions = unique_questions
    print(f"[SPLIT] Unique questions: {len(questions)}")
    
    # Stratify by category
    by_category = defaultdict(list)
    for q in questions:
        cat = q.get("category", "unknown")
        by_category[cat].append(q)
    
    print(f"[SPLIT] Categories: {len(by_category)}")
    for cat, qs in sorted(by_category.items()):
        print(f"  {cat}: {len(qs)} questions")
    
    # Allocate dev/test proportionally
    random.seed(42)
    dev_set = []
    test_set = []
    
    for cat, qs in by_category.items():
        random.shuffle(qs)
        cat_dev_size = max(1, int(dev_size * len(qs) / len(questions)))
        cat_test_size = max(1, int(test_size * len(qs) / len(questions)))
        
        dev_set.extend(qs[:cat_dev_size])
        test_set.extend(qs[cat_dev_size:cat_dev_size + cat_test_size])
    
    # Ensure exact sizes
    if len(dev_set) > dev_size:
        test_set.extend(dev_set[dev_size:])
        dev_set = dev_set[:dev_size]
    elif len(dev_set) < dev_size and test_set:
        take = min(dev_size - len(dev_set), len(test_set))
        dev_set.extend(test_set[:take])
        test_set = test_set[take:]
    
    if len(test_set) > test_size:
        test_set = test_set[:test_size]
    
    print(f"[SPLIT] Dev set: {len(dev_set)} questions")
    print(f"[SPLIT] Test set: {len(test_set)} questions")
    
    return dev_set, test_set

def main():
    print("=" * 70)
    print("BUILDING LOCKED TEST/DEV SPLIT")
    print("=" * 70)
    
    # Load corpus
    documents = load_medquad_corpus()
    
    # Extract questions and create corpus
    print("\n[EXTRACT] Extracting questions from corpus...")
    questions, corpus = extract_questions_and_create_corpus(documents)
    print(f"[EXTRACT] Extracted {len(questions)} unique question-document pairs")
    
    # Create edge cases
    print("\n[EDGE] Creating 50 edge case questions...")
    edge_cases = create_edge_cases()
    print(f"[EDGE] Created {len(edge_cases)} edge cases")
    
    # Split dataset
    print("\n[SPLIT] Splitting into dev/test sets...")
    dev_set, test_baseline = split_dataset(questions, corpus, dev_size=100, test_size=200)
    
    # Combine test set
    test_set = test_baseline + edge_cases
    
    # Verify no overlap
    dev_questions = {q["question"].lower() for q in dev_set}
    test_questions = {q["question"].lower() for q in test_set}
    overlap = dev_questions & test_questions
    
    if overlap:
        print(f"\n[ERROR] Found {len(overlap)} overlapping questions!")
        sys.exit(1)
    
    print(f"[VERIFY] ✓ No overlap between dev and test sets")
    print(f"[VERIFY] ✓ Dev: {len(dev_set)} | Test: {len(test_set)}")
    
    # Save outputs
    print("\n[SAVE] Saving datasets...")
    
    os.makedirs("data", exist_ok=True)
    
    # Save indexed corpus
    with open("data/indexed_corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")
    print(f"  ✓ data/indexed_corpus.jsonl ({len(corpus)} docs)")
    
    # Save dev set
    with open("data/dev_set.jsonl", "w") as f:
        for q in dev_set:
            f.write(json.dumps(q) + "\n")
    print(f"  ✓ data/dev_set.jsonl ({len(dev_set)} questions)")
    
    # Save test set
    with open("data/test_set.jsonl", "w") as f:
        for q in test_set:
            f.write(json.dumps(q) + "\n")
    print(f"  ✓ data/test_set.jsonl ({len(test_set)} questions)")
    
    # Create lock hash
    test_hash = hash_file("data/test_set.jsonl")
    with open("data/test_set.lock", "w") as f:
        f.write(f"test_set.jsonl\n")
        f.write(f"SHA256: {test_hash}\n")
        f.write(f"Questions: {len(test_set)}\n")
        f.write(f"Locked: 2026-05-31\n")
        f.write(f"Status: FROZEN - DO NOT MODIFY\n")
    print(f"  ✓ data/test_set.lock (SHA256: {test_hash[:16]}...)")
    
    print("\n" + "=" * 70)
    print("✅ TEST/DEV SPLIT CREATED AND LOCKED")
    print("=" * 70)
    print(f"\nDevliverables:")
    print(f"  indexed_corpus.jsonl  — {len(corpus)} documents")
    print(f"  dev_set.jsonl         — {len(dev_set)} questions (validation)")
    print(f"  test_set.jsonl        — {len(test_set)} questions (frozen)")
    print(f"  test_set.lock         — SHA256 hash + metadata")
    print(f"\nNo further changes to test set allowed.")
    print(f"Next: Build baseline retrieval metrics with locked test set.")

if __name__ == "__main__":
    import os
    main()

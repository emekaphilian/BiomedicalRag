#!/usr/bin/env python
"""
Embeddings Test Script - Run this file to verify everything works
Usage: python test_embeddings_full.py
"""

import sys
import os

print("\n" + "="*70)
print("BLOCK 1 VERIFICATION TEST")
print("="*70 + "\n")

# Test 1: Directory structure
print("[1/5] Checking directory structure...")
required_dirs = [
    "rag", "evaluation", "app", "data", "configs", "tests"
]
dirs_ok = True
for dir_name in required_dirs:
    if os.path.isdir(dir_name):
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ {dir_name}/ (MISSING)")
        dirs_ok = False

if not dirs_ok:
    print("\n✗ Directory structure incomplete!")
    sys.exit(1)

# Test 2: Module files exist
print("\n[2/5] Checking core module files...")
required_files = [
    "rag/__init__.py",
    "rag/embeddings.py",
    "rag/vectorstore.py",
    "rag/pipeline.py",
    "evaluation/__init__.py",
    "evaluation/retrieval_metrics.py",
    "configs/config.yaml",
    "requirements.txt"
]
files_ok = True
for file_path in required_files:
    if os.path.isfile(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} (MISSING)")
        files_ok = False

if not files_ok:
    print("\n✗ Some required files are missing!")
    sys.exit(1)

# Test 3: Core imports
print("\n[3/5] Testing core module imports...")
try:
    from rag import embeddings, vectorstore, pipeline
    print("  ✓ rag.embeddings imported")
    print("  ✓ rag.vectorstore imported")
    print("  ✓ rag.pipeline imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

try:
    from evaluation import retrieval_metrics, generation_metrics
    print("  ✓ evaluation.retrieval_metrics imported")
    print("  ✓ evaluation.generation_metrics imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 4: Sentence-transformers available
print("\n[4/5] Checking sentence-transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print("  ✓ sentence-transformers available")
except ImportError:
    print("  ✗ sentence-transformers NOT installed")
    print("     Run: pip install sentence-transformers")
    sys.exit(1)

# Test 5: Embeddings generation (CRITICAL)
print("\n[5/5] Testing embeddings generation (CRITICAL)...")
print("  Loading model: all-MiniLM-L6-v2...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  ✓ Model loaded successfully")
    
    # Generate test embeddings
    test_sentences = [
        "Hypertension treatment",
        "Diabetes management",
        "COVID-19 vaccine"
    ]
    print("  Generating test embeddings...")
    embeddings = model.encode(test_sentences, show_progress_bar=False)
    print(f"  ✓ Generated {len(embeddings)} embeddings")
    
    # Verify dimensions
    embedding_dim = embeddings[0].shape[0]
    print(f"  ✓ Embedding dimension: {embedding_dim} (expected: 384)")
    
    if embedding_dim != 384:
        print(f"  ✗ WRONG dimension! Expected 384, got {embedding_dim}")
        sys.exit(1)
    
    # Test similarity
    similarity_score = model.similarity(embeddings[0], embeddings[1])
    print(f"  ✓ Similarity computation: {float(similarity_score[0][0]):.4f}")
    
except Exception as e:
    print(f"  ✗ Embeddings test FAILED: {e}")
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✓✓✓ ALL TESTS PASSED - BLOCK 1 COMPLETE ✓✓✓")
print("="*70)
print("\nYour biomedical RAG system is ready for Phase 2!")
print("Next: Implement retrieval grounding and safety constraints")
print("="*70 + "\n")

sys.exit(0)

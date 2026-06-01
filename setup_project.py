"""
Setup script for Biomedical RAG System
Creates project structure and verifies environment
Run with: python setup_project.py
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Create required directory structure."""
    dirs = [
        "rag",
        "evaluation",
        "data",
        "data/raw",
        "data/processed",
        "data/index",
        "configs",
        "tests",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory: {dir_path}/")
    
    return len(dirs)

def verify_environment():
    """Verify Python and required packages."""
    print("\n" + "="*60)
    print("ENVIRONMENT VERIFICATION")
    print("="*60)
    
    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✓ Python: {python_version}")
    
    # Check if venv exists
    venv_path = Path("venv")
    if venv_path.exists():
        print(f"✓ Virtual environment: venv/")
    else:
        print("⚠ Virtual environment not found. Create with: python -m venv venv")
    
    return True

def verify_structure():
    """Verify project structure."""
    print("\n" + "="*60)
    print("PROJECT STRUCTURE VERIFICATION")
    print("="*60)
    
    required_files = {
        "rag/__init__.py": "RAG module",
        "rag/pipeline.py": "RAG pipeline",
        "rag/embeddings.py": "Embeddings provider",
        "rag/vectorstore.py": "Vector store",
        "rag/retriever.py": "Document retriever",
        "rag/generator.py": "Answer generator",
        "evaluation/__init__.py": "Evaluation module",
        "evaluation/retrieval_metrics.py": "Retrieval metrics",
        "evaluation/generation_metrics.py": "Generation metrics",
        "app/api.py": "FastAPI application",
        "app/streamlit_app.py": "Streamlit application",
        "configs/config.yaml": "Configuration file",
        "requirements.txt": "Dependencies",
        "README.md": "Documentation",
    }
    
    status = {
        "found": 0,
        "missing": 0,
    }
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✓ {file_path:<40} ({description})")
            status["found"] += 1
        else:
            print(f"✗ {file_path:<40} ({description})")
            status["missing"] += 1
    
    return status

def test_imports():
    """Test core module imports."""
    print("\n" + "="*60)
    print("IMPORT VERIFICATION")
    print("="*60)
    
    test_imports_list = [
        ("rag", "RAG module"),
        ("rag.embeddings", "Embeddings"),
        ("rag.vectorstore", "Vector store"),
        ("rag.pipeline", "Pipeline"),
        ("evaluation", "Evaluation module"),
    ]
    
    success_count = 0
    for module_name, description in test_imports_list:
        try:
            __import__(module_name)
            print(f"✓ {module_name:<30} ({description})")
            success_count += 1
        except ImportError as e:
            print(f"✗ {module_name:<30} ({description})")
            print(f"  Error: {str(e)[:60]}")
    
    return success_count == len(test_imports_list)

def test_embeddings():
    """Test embeddings initialization (requires sentence-transformers)."""
    print("\n" + "="*60)
    print("EMBEDDINGS TEST")
    print("="*60)
    
    try:
        from rag.embeddings import EmbeddingsProvider
        print("✓ EmbeddingsProvider imported")
        
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("✓ sentence-transformers available")
            
            # Try to load model
            print("  Loading model: all-MiniLM-L6-v2 (first time may download)...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(["Hello world"])
            print(f"✓ Model loaded successfully")
            print(f"  Embedding dimension: {embeddings[0].shape[0]}")
            return True
        except ImportError:
            print("⚠ sentence-transformers not installed")
            print("  Install with: pip install -r requirements.txt")
            return False
    except Exception as e:
        print(f"✗ Embeddings test failed: {str(e)[:60]}")
        return False

def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("BIOMEDICAL RAG SYSTEM - PROJECT SETUP")
    print("="*60 + "\n")
    
    # Create directories
    print("Creating directories...")
    dir_count = create_directories()
    print(f"✓ {dir_count} directories created\n")
    
    # Verify environment
    verify_environment()
    
    # Verify structure
    structure_status = verify_structure()
    print(f"\n✓ Found: {structure_status['found']}")
    print(f"✗ Missing: {structure_status['missing']}")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test embeddings
    embeddings_ok = test_embeddings()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if structure_status["missing"] == 0 and imports_ok:
        print("✓ PROJECT STRUCTURE: READY")
    else:
        print("⚠ PROJECT STRUCTURE: INCOMPLETE")
    
    if embeddings_ok:
        print("✓ EMBEDDINGS: READY")
    else:
        print("⚠ EMBEDDINGS: INSTALL DEPENDENCIES")
        print("  Run: pip install -r requirements.txt")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Install dependencies (if not already done):
   pip install -r requirements.txt

2. Verify embeddings work:
   python setup_project.py

3. Run tests:
   pytest tests/ -v

4. Start development:
   - Streamlit: streamlit run app/streamlit_app.py
   - FastAPI:  python app/api.py
    """)

if __name__ == "__main__":
    main()

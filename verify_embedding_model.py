"""Verify BioLORD embedding model is loaded"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("BIOLORD EMBEDDING MODEL VERIFICATION")
print("=" * 70)

# Check vectorstore configuration
print("\n1. Checking BiomedicalVectorStore configuration...")
with open("rag/vectorstore.py") as f:
    content = f.read()
    if 'BioLORD' in content or 'biolord' in content.lower():
        print("   ✅ BioLORD found in vectorstore.py")
    else:
        print("   ⚠️  BioLORD NOT in vectorstore.py")
    
    # Show the actual model being used
    for line in content.split('\n'):
        if 'SentenceTransformer' in line and '(' in line:
            print(f"   Current: {line.strip()}")

# Check app services
print("\n2. Checking app/services.py...")
try:
    import app.services as services
    if hasattr(services, 'DEFAULT_EMBEDDING_MODEL'):
        model = services.DEFAULT_EMBEDDING_MODEL
        print(f"   DEFAULT_EMBEDDING_MODEL: {model}")
    else:
        print("   ⚠️  DEFAULT_EMBEDDING_MODEL not found in app.services")
except Exception as e:
    print(f"   ⚠️  Error importing app.services: {e}")

# Try loading actual embedding model
print("\n3. Testing actual model load...")
try:
    from sentence_transformers import SentenceTransformer
    print("   Loading 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"   ✓ Loaded: {model.modules()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
To implement BioLORD as per proposal:

1. Edit rag/vectorstore.py line 13:
   FROM: self.model = SentenceTransformer("all-MiniLM-L6-v2")
   TO:   self.model = SentenceTransformer("BioLORD-2023")

2. First time load will download BioLORD weights (~700MB)

3. Re-embed corpus with:
   python -c "from rag.vectorstore import BiomedicalVectorStore; 
              vs = BiomedicalVectorStore(); 
              print(f'Model: {vs.model.modules()}')"

4. Verify:
   python verify_corpus.py
""")

"""Quick test of retriever response structure"""
import sys
sys.path.insert(0, '.')

from rag.ingestion import load_medquad_dataset
from rag.vectorstore import BiomedicalVectorStore
from rag.hybrid_retriever import HybridRetriever

print('Loading documents...')
docs = load_medquad_dataset('data/raw')
print(f'Loaded {len(docs)} documents')

print('\nBuilding vectorstore with first 100 docs...')
vs = BiomedicalVectorStore()
vs.add_documents(docs[:100])

print('Creating retriever...')
retriever = HybridRetriever(vectorstore=vs, documents=docs[:100])

print('\nTesting retrieve...')
result = retriever.retrieve('What are the stages of testicular cancer?', k=5)
print(f'Response keys: {list(result.keys())}')

for key in result.keys():
    val = result[key]
    if isinstance(val, list):
        print(f'  {key}: list of {len(val)} items')
        if val:
            print(f'    First item type: {type(val[0])}')
            print(f'    First item: {val[0]}')
    else:
        print(f'  {key}: {type(val).__name__} = {val}')

#!/usr/bin/env python
"""
Stabilization Phase - Task 1: Real LLM Integration with Cohere
Test biomedical grounding with Cohere Command-R model
"""

import os
import sys

os.chdir(r"f:\Users\phili\Documents\Projects\LLM-powered-document-Q&A-system-RAG\rag-qa")
sys.path.insert(0, ".")

# Check for API key
if not os.getenv("COHERE_API_KEY"):
    print("=" * 80)
    print("ERROR: COHERE_API_KEY not set")
    print("=" * 80)
    print("\nSet it in PowerShell:")
    print('  $env:COHERE_API_KEY="your-cohere-key-here"')
    print("\nOr in Bash:")
    print("  export COHERE_API_KEY='your-cohere-key-here'")
    print("\nGet a free key at: https://dashboard.cohere.com/")
    sys.exit(1)

print("Importing modules...")
try:
    from rag.ingestion import load_medquad_dataset
    from rag.vectorstore import BiomedicalVectorStore
    from rag.hybrid_retriever import HybridRetriever
    from rag.compression import ContextCompressor
    from rag.prompt_constructor import GroundedPromptConstructor
    from rag.generator_cohere import CohereGenerator
    from rag.post_validation import PostGenerationValidator
    print("[OK] All modules imported successfully")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STABILIZATION PHASE - COHERE LLM INTEGRATION TEST")
print("=" * 80)

try:
    # Step 1: Load data
    print("\n[1/5] Loading biomedical dataset...")
    docs = load_medquad_dataset(dataset_path="data/raw")
    print(f"[OK] Loaded {len(docs)} documents")
    
    # Step 2: Build indexes
    print("\n[2/5] Building retrieval indexes...")
    vectorstore = BiomedicalVectorStore()
    vectorstore.add_documents(docs)
    print(f"[OK] FAISS index built with {len(docs)} documents indexed")
    
    # Step 3: Initialize retrieval pipeline
    print("\n[3/5] Initializing hybrid retrieval...")
    hybrid_retriever = HybridRetriever(vectorstore=vectorstore, documents=docs)
    print("[OK] Retrieval pipeline ready")
    
    # Step 4: Initialize LLM
    print("\n[4/5] Initializing Cohere generator...")
    generator = CohereGenerator(model="command-nightly")
    compressor = ContextCompressor()
    prompt_constructor = GroundedPromptConstructor()
    validator = PostGenerationValidator()
    print("[OK] Cohere generator initialized")
    
    # Step 5: Test queries
    print("\n[5/5] Running grounding validation tests...")
    print("=" * 80)
    
    test_queries = [
        "What are the main symptoms of diabetes?",
        "What causes Parkinson's disease?",
        "What is leukemia?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/3] Query: {query}")
        print("-" * 80)
        
        try:
            # Retrieve
            retrieval_response = hybrid_retriever.retrieve(query, k=5)
            
            # Get fused results (hybrid fusion of dense + sparse)
            fused_results = retrieval_response["fused_results"]
            print(f"  Retrieved {len(fused_results)} fused results")
            
            # fused_results are strings directly, format as context
            context_items = []
            for idx, chunk in enumerate(fused_results[:5], 1):
                # Each chunk is already a string from the fusion
                context_items.append({
                    "id": idx,
                    "source": "MedQuAD",
                    "content": str(chunk)[:500],
                    "rerank_score": 0.9
                })
            
            # Compress format
            compressed = {
                "context": context_items,
                "sources": ["MedQuAD"],
                "evidence_count": len(context_items),
                "formatted_text": "\n\n".join([f"[Evidence {i}] {c['content']}" for i, c in enumerate(context_items, 1)])
            }
            print(f"  Formatted {len(context_items)} chunks for prompt")
            
            # Construct prompt
            prompt = prompt_constructor.construct(query=query, compressed_context=compressed)
            evidence_count = compressed['evidence_count']
            
            # Generate with Cohere
            answer = generator.generate(
                query=query,
                context=compressed['formatted_text']
            )
            print(f"  Generated answer ({len(answer)} chars)")
            
            # Validate
            generation_result = {
                "success": True,
                "answer": answer,
                "error": None
            }
            
            validation = validator.validate(
                generation_result=generation_result,
                evidence_count=evidence_count
            )
            
            print(f"  Validation: {'PASS' if validation['valid'] else 'FAIL'}")
            if not validation['valid']:
                print(f"  Issues: {validation['issues']}")
            
            # Show answer preview
            preview = answer[:200].replace("\n", " ")
            print(f"\n  Answer preview:")
            print(f"  {preview}...")
            
            results.append({
                "query": query,
                "valid": validation['valid'],
                "answer": answer
            })
            
            if validation['valid']:
                print("\n  [OK] GROUNDING VALIDATED")
            else:
                print(f"\n  [WARN] Validation issues: {', '.join(validation['issues'])}")
        
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results.append({"query": query, "valid": False, "error": str(e)})
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.get("valid"))
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for i, result in enumerate(results, 1):
        status = "PASS" if result.get("valid") else "FAIL"
        print(f"  [{i}] {result['query'][:50]}... - {status}")
    
    if passed == total:
        print("\n[SUCCESS] All grounding tests passed!")
        print("Cohere integration working correctly.")
        sys.exit(0)
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Test cancelled by user")
    sys.exit(1)

except Exception as e:
    print(f"\n[FATAL ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

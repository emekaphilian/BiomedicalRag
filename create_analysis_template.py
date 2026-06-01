"""
Manual retrieval analysis template.

After running stabilization_task3c_error_analysis.py:

1. Open retrieval_debug.jsonl
2. For each query, examine the retrieved chunks
3. Ask: "Was the relevant information present in the top 5?"
4. Fill in this analysis with your findings
"""

import csv
from pathlib import Path


def create_analysis_template():
    """Create CSV template for manual analysis"""

    # Load queries
    from stabilization_task3c_error_analysis import EVALUATION_QUERIES

    all_queries = []
    for cat, queries in EVALUATION_QUERIES.items():
        all_queries.extend([(cat, q) for q in queries])

    # Create CSV
    with open("retrieval_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Query_Num",
                "Category",
                "Query",
                "Expected_Topic",
                "Was_Relevant_Info_Present",
                "Error_Type",
                "Details",
                "Suggested_Fix",
            ],
        )
        writer.writeheader()

        for idx, (cat, query) in enumerate(all_queries, 1):
            # Extract expected topic from query
            if "symptoms" in query.lower() or "causes" in query.lower():
                expected_topic = "Disease/Symptom"
            elif "how does" in query.lower() or "how" in query.lower():
                expected_topic = "Mechanism"
            elif "related" in query.lower() or "complications" in query.lower():
                expected_topic = "Relationship"
            else:
                expected_topic = "Other"

            writer.writerow({
                "Query_Num": idx,
                "Category": cat,
                "Query": query,
                "Expected_Topic": expected_topic,
                "Was_Relevant_Info_Present": "?",  # YOU FILL IN: YES/NO
                "Error_Type": "",  # YOU FILL IN: See error types below
                "Details": "",  # YOU FILL IN: Your observations
                "Suggested_Fix": "",  # YOU FILL IN: Your recommendation
            })

    print("✓ Created retrieval_analysis.csv")
    print("\nError Types (fill in 'Error_Type' column):")
    print("  - Embedding_mismatch: Query and relevant doc use different vocabulary")
    print("  - Keyword_mismatch: Technical term not in retrieval query")
    print("  - Chunk_too_small: Relevant info exists but in different chunk")
    print("  - Chunk_too_large: Too much irrelevant context in chunk")
    print("  - RRF_ranking: Relevant chunk ranked low (BM25+Dense mismatch)")
    print("  - Reranker_issue: Cross-encoder ranked relevant chunk down")
    print("  - Dataset_gap: Answer not in corpus at all")
    print("  - Ambiguous_query: Query has multiple interpretations")
    print("  - None: Successfully retrieved")
    print("\nInstructions:")
    print("1. Open retrieval_debug.jsonl")
    print("2. For each query, check top 5 chunks")
    print("3. For each chunk, check if has_relevant_keyword=true")
    print("4. If yes, fill Was_Relevant_Info_Present=YES")
    print("5. If no info in top 5, fill Was_Relevant_Info_Present=NO")
    print("6. Determine error type based on why it failed")
    print("7. Fill 'Details' with specific observations")
    print("8. Suggest fix if obvious")


if __name__ == "__main__":
    create_analysis_template()

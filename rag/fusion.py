def reciprocal_rank_fusion(dense_results, sparse_results, k=60):

    scores = {}

    for rank, result in enumerate(dense_results):

        content = result.content

        scores[content] = scores.get(content, 0) + 1 / (k + rank + 1)

    for rank, result in enumerate(sparse_results):

        content = result["document"].content

        scores[content] = scores.get(content, 0) + 1 / (k + rank + 1)

    reranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked

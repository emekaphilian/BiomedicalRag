from dashboard.rag_audit import audit_answer, retrieval_summary, summarize_audits


def test_audit_reports_sentence_support_and_invalid_citations():
    audit = audit_answer(
        "Dopamine neurons affect movement [1]. There are 19 adverse clinical outcomes [3].",
        ["Dopamine neurons affect movement."],
    )
    assert audit["faithfulness"] == 0.5
    assert audit["invalid_citations"] == [3]
    assert audit["flagged_sentences"][0]["unsupported_numbers"] == ["19"]


def test_audit_summaries_handle_no_data_and_metric_k_variants():
    assert audit_answer("Thanks.", [])["faithfulness"] is None
    assert summarize_audits([])["faithfulness"] is None
    summary = retrieval_summary([{"metrics": {"recall@5": 0.5, "precision@5": 0.25, "ndcg@5": 0.4, "mrr": 0.6}}])
    assert summary == {"runs": 1, "recall": 0.5, "precision": 0.25, "ndcg": 0.4, "mrr": 0.6}

import json

from dashboard.rag_audit import audit_answer, retrieval_summary, summarize_audits
from rag.post_validation import PostGenerationValidator
from rag.evidence_relevance import classify_evidence


def test_audit_reports_sentence_support_and_invalid_citations():
    audit = audit_answer(
        "Dopamine neurons affect movement [1]. There are 19 adverse clinical outcomes [3].",
        ["Dopamine neurons affect movement."],
    )
    assert audit["faithfulness"] == 0.5
    assert audit["invalid_citations"] == [3]
    assert audit["flagged_sentences"][0]["unsupported_numbers"] == ["19"]


def test_audit_recognizes_evidence_citation_format_and_separates_coverage():
    audit = audit_answer(
        "Dopamine neurons affect movement [Evidence 1].",
        ["Dopamine neurons affect movement."],
    )
    assert audit["faithfulness"] == 1.0
    assert audit["attribution_coverage"] == 1.0
    assert audit["evidence_trace_coverage"] == 1.0
    assert audit["citation_support_coverage"] == 1.0
    assert audit["invalid_citations"] == []


def test_audit_normalizes_common_biomedical_plain_language_equivalents():
    audit = audit_answer(
        "Diabetes is characterized by elevated blood sugar [Evidence 1].",
        ["Diabetes is a condition in which blood glucose levels are too high."],
    )
    assert audit["faithfulness"] == 1.0
    assert audit["citation_support_coverage"] == 1.0


def test_claim_validation_retains_citation_and_support_for_plain_language_equivalent():
    result = PostGenerationValidator().validate(
        {"success": True, "answer": "Diabetes is characterized by elevated blood sugar [Evidence 1]."},
        evidence_count=1,
        evidence_texts=["Diabetes is a condition in which blood glucose levels are too high."],
        evidence_records=[{"evidence_id": "evidence-1", "metadata": {"source": "MedlinePlus"}}],
    )
    assert result["valid"] is True
    assert result["claim_support"][0]["support_status"] == "supported"
    assert result["claim_support"][0]["evidence_ids"] == ["evidence-1"]


def test_validation_logs_inputs_score_and_failure_reason(caplog):
    answer = "A tumor is a mass formed by extra cells [Evidence 1]."
    passage = "A tumor is a mass of extra cells."
    record = {
        "content": passage,
        "metadata": {"source": "MedlinePlus"},
        "evidence_classification": {"category": "direct", "intent_match": True},
    }

    with caplog.at_level("INFO", logger="rag.post_validation"):
        result = PostGenerationValidator().validate(
            {"success": True, "answer": answer}, 1, [passage], [record],
        )

    event = next(record.message for record in caplog.records if record.message.startswith("rag_post_generation_validation="))
    payload = json.loads(event.split("=", 1)[1])
    assert payload["retrieved_chunks"] == [passage]
    assert payload["generated_answer"] == answer
    assert payload["validation"]["diagnostics"]["method"] == "lexical_overlap_fallback"
    assert payload["failure_reason"] == result["reason"]
    assert payload["thresholds"]["minimum_supporting_chunks"] == 1


def test_empty_generation_is_not_accepted_as_valid():
    result = PostGenerationValidator().validate(
        {"success": True, "answer": "   "},
        evidence_count=1,
        evidence_texts=["Dopamine is involved in movement."],
    )

    assert result["valid"] is False
    assert "Generated answer is empty" in result["issues"]


def test_risk_association_cannot_validate_a_direct_cause_claim():
    query = "What causes hypertension?"
    passage = "The risk of high blood pressure increases as you get older."
    record = {
        "evidence_id": "ev-risk",
        "content": passage,
        "metadata": {"source": "MedlinePlus", "title": "High Blood Pressure"},
        "evidence_classification": classify_evidence(query, passage, {"title": "High Blood Pressure"}),
    }
    result = PostGenerationValidator().validate(
        {"success": True, "answer": "Aging directly causes hypertension [Evidence 1]."},
        evidence_count=1,
        evidence_texts=[passage],
        evidence_records=[record],
    )
    assert result["valid"] is False
    assert result["claim_support"][0]["support_status"] == "unsupported"
    assert result["claim_support"][0]["evidence_ids"] == []


def test_audit_summaries_handle_no_data_and_metric_k_variants():
    assert audit_answer("Thanks.", [])["faithfulness"] is None
    assert summarize_audits([])["faithfulness"] is None
    summary = retrieval_summary([{"metrics": {"recall@5": 0.5, "precision@5": 0.25, "ndcg@5": 0.4, "mrr": 0.6}}])
    assert summary == {
        "runs": 1,
        "recall@5": 0.5,
        "precision@5": 0.25,
        "ndcg@5": 0.4,
        "mrr": 0.6,
    }


def test_retrieval_summary_keeps_different_cutoffs_separate():
    summary = retrieval_summary([
        {"metrics": {"recall@3": 0.5, "precision@3": 0.3, "ndcg@3": 0.4, "mrr": 0.6}},
        {"metrics": {"recall@5": 1.0, "precision@5": 0.2, "ndcg@5": 0.8, "mrr": 0.8}},
    ])
    assert summary["recall@3"] == 0.5
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == 0.7

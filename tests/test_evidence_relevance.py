from rag.evidence_relevance import classify_evidence, classify_evidence_set, interpret_query


def test_query_intent_is_general_and_expands_requested_evidence_type():
    intent = interpret_query("WHAT ARE common symptoms of diabetes?!")
    assert intent.information_type == "symptoms"
    assert "diabet" in intent.subject_terms
    assert "manifestation" in intent.retrieval_query


def test_diabetes_symptoms_distinguishes_direct_related_and_complication_evidence():
    direct = classify_evidence(
        "What are common symptoms of diabetes?",
        "Common symptoms of diabetes include increased thirst and frequent urination.",
    )
    prediabetes = classify_evidence(
        "What are common symptoms of diabetes?",
        "Prediabetes occurs when blood glucose is higher than normal.",
    )
    eye_complication = classify_evidence(
        "What are common symptoms of diabetes?",
        "Diabetic eye problems include diabetic retinopathy and vision loss.",
        {"title": "Diabetic eye problems"},
    )
    irrelevant = classify_evidence(
        "What are common symptoms of diabetes?",
        "Parkinson's disease affects dopamine-producing cells.",
    )

    assert direct["category"] == "direct"
    assert prediabetes["category"] == "related_indirect"
    assert eye_complication["category"] == "complication"
    assert irrelevant["category"] == "irrelevant"


def test_general_intent_examples_classify_as_direct():
    cases = [
        ("What causes diabetes?", "Diabetes can be caused by genetic and environmental factors."),
        ("What is diabetes?", "Diabetes is a chronic condition affecting blood glucose."),
        ("How does Parkinson's disease affect dopamine?", "Parkinson's disease affects dopamine-producing neurons."),
        ("What are Ebola symptoms?", "Symptoms of Ebola include fever and fatigue."),
        ("How does insulin work?", "Insulin helps regulate glucose and allows cells to use it."),
        ("What is dopamine?", "Dopamine is a neurotransmitter involved in brain signaling."),
    ]
    for query, passage in cases:
        assert classify_evidence(query, passage)["category"] == "direct", query


def test_section_heading_and_title_are_used_as_context_but_not_as_sufficient_proof():
    result = classify_evidence(
        "What are common symptoms of diabetes?",
        "This section discusses care and follow-up.",
        {"title": "Diabetes", "section": "Overview"},
    )
    assert result["category"] == "related_indirect"
    assert result["intent_match"] is False


def test_explicit_opposing_statements_are_flagged_as_contradictory():
    query = "What does diabetes do to kidney risk?"
    evidence = [
        {"content": "Diabetes increases risk of kidney disease."},
        {"content": "Diabetes decreases risk of kidney disease."},
    ]
    results = classify_evidence_set(query, evidence)
    assert [result["category"] for result in results] == ["contradictory", "contradictory"]

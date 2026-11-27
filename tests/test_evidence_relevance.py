from rag.evidence_relevance import assess_evidence_coverage, classify_evidence, classify_evidence_set, interpret_query


def test_query_intent_is_general_and_expands_requested_evidence_type():
    intent = interpret_query("WHAT ARE common symptoms of diabetes?!")
    assert intent.information_type == "symptoms"
    assert "diabet" in intent.subject_terms
    assert "manifestation" in intent.retrieval_query


def test_treatment_query_expansion_stays_specific_to_the_subject():
    intent = interpret_query("Summarize Alzheimer's therapies")

    assert intent.information_type == "treatments"
    assert intent.subject_terms == ("alzheimer",)
    assert intent.retrieval_query == "alzheimer disease treatment"
    assert "symptomatic" not in intent.retrieval_query
    assert "disease-modifying" not in intent.retrieval_query


def test_long_treatment_question_rewrites_to_its_subject_and_intent():
    intent = interpret_query(
        "What evidence supports current understanding of Alzheimer's therapies work?"
    )

    assert intent.retrieval_query == "alzheimer disease treatment"


def test_alzheimers_staging_is_not_mislabeled_as_a_complication():
    result = classify_evidence(
        "Summarize Alzheimer's therapies",
        "Alzheimer's disease clinical staging describes disease progression and memory problems.",
        {"title": "Alzheimer's Disease"},
    )

    assert result["category"] == "supporting_contextual"
    assert result["matched_subject_terms"] == ["alzheimer"]


def test_treatment_evidence_is_direct_and_other_disease_therapy_is_irrelevant():
    query = "Summarize Alzheimer's therapies"
    treatment = classify_evidence(
        query,
        "Treatments for Alzheimer's disease may include medicines that help manage symptoms.",
        {"title": "Alzheimer's Disease"},
    )
    unrelated = classify_evidence(
        query,
        "Cancer alternative therapies are used for cancer care.",
        {"title": "Cancer Alternative Therapies"},
    )

    assert treatment["category"] == "direct"
    assert unrelated["category"] == "irrelevant"


def test_medication_history_does_not_count_as_treatment_evidence():
    result = classify_evidence(
        "Summarize Alzheimer's therapies",
        "To assess memory problems, review the person's medical history and any medicines they are taking.",
        {"title": "Alzheimer's Disease"},
    )

    assert result["category"] == "related_indirect"


def test_definition_remains_direct_when_passage_mentions_disease_progression():
    result = classify_evidence(
        "What is diabetes?",
        "Diabetes is a disease in which blood glucose is too high. Disease progression can lead to serious health conditions.",
        {"title": "Diabetes"},
    )

    assert result["category"] == "direct"


def test_hypertension_plain_language_definition_and_unknown_cause_are_direct():
    definition = classify_evidence(
        "What is hypertension?",
        "High blood pressure means the force of blood against artery walls is too high.",
        {"title": "High Blood Pressure"},
    )
    unknown_cause = classify_evidence(
        "What causes hypertension?",
        "The exact cause of high blood pressure is not known.",
        {"title": "High Blood Pressure"},
    )

    assert definition["category"] == "direct"
    assert unknown_cause["category"] == "direct"


def test_requested_definition_and_cause_examples_are_classified_as_direct():
    cases = [
        ("What is hypertension?", "High blood pressure means the force of blood against artery walls is too high."),
        ("What causes hypertension?", "Secondary hypertension is caused by another medical condition or medicine."),
        ("What is a tumor?", "A tumor is a mass formed by extra cells that grow when normal cell growth and death processes go wrong."),
        ("What causes a tumor?", "A tumor is a mass formed by extra cells that grow when normal cell growth and death processes go wrong."),
    ]
    for query, passage in cases:
        assert classify_evidence(query, passage, {"title": "MedlinePlus"})["category"] == "direct", query


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


def test_hypertension_cause_query_separates_systemic_and_pulmonary_conditions():
    systemic = classify_evidence(
        "What causes hypertension?",
        "Primary high blood pressure develops over time. Secondary hypertension is caused by another medical condition or medicine.",
        {"title": "High Blood Pressure"},
    )
    pulmonary = classify_evidence(
        "What causes hypertension?",
        "Pulmonary hypertension affects the arteries in the lungs.",
        {"title": "Pulmonary Hypertension"},
    )

    assert systemic["category"] == "direct"
    assert systemic["entity_match_type"] in {"exact_concept", "synonym"}
    assert pulmonary["category"] == "related_indirect"
    assert pulmonary["entity_match_type"] == "related_condition"


def test_risk_association_does_not_count_as_direct_cause_evidence():
    result = classify_evidence(
        "What causes hypertension?",
        "The risk of high blood pressure increases as you get older.",
        {"title": "High Blood Pressure"},
    )
    assert result["category"] == "supporting_contextual"
    assert result["intent_match"] is False


def test_cancer_mutation_mechanism_is_direct_cause_evidence_but_risk_factors_are_not():
    mechanism = classify_evidence(
        "What is the cause of cancer?",
        "Cancer is a genetic disease caused by changes (mutations) in genes that affect cell function.",
        {"title": "Cancer"},
    )
    risk_factors = classify_evidence(
        "What is the cause of cancer?",
        "Risk factors for cancer include ultraviolet radiation, smoking, diet, physical inactivity, and aging.",
        {"title": "Cancer"},
    )

    assert mechanism["category"] == "direct"
    assert risk_factors["category"] == "supporting_contextual"


def test_comparison_coverage_is_partial_when_only_one_side_is_retrieved():
    evidence = [{
        "evidence_id": "ev-type1",
        "content": "Type 1 diabetes occurs when the body does not make insulin.",
        "metadata": {"title": "Type 1 Diabetes"},
    }]
    evidence[0]["evidence_classification"] = classify_evidence(
        "Compare type 1 vs type 2 diabetes.", evidence[0]["content"], evidence[0]["metadata"],
    )
    coverage = assess_evidence_coverage("Compare type 1 vs type 2 diabetes.", evidence)
    assert coverage["status"] == "partial"
    assert coverage["coverage_score"] == 0.5


def test_generic_diabetes_background_is_not_direct_evidence_for_a_comparison():
    query = "Compare type 1 vs type 2 diabetes."
    evidence = [
        {
            "evidence_id": "ev-general",
            "content": "Diabetes is a disease in which blood glucose levels are too high.",
            "metadata": {"title": "Diabetes"},
        },
        {
            "evidence_id": "ev-type1",
            "content": "Type 1 diabetes occurs when the body does not make insulin.",
            "metadata": {"title": "Type 1 Diabetes"},
        },
    ]
    for item in evidence:
        item["evidence_classification"] = classify_evidence(
            query, item["content"], item["metadata"],
        )

    coverage = assess_evidence_coverage(query, evidence)

    assert evidence[0]["evidence_classification"]["category"] == "related_indirect"
    assert evidence[0]["evidence_classification"]["comparison_entities_supported"] == []
    assert coverage["status"] == "partial"
    assert coverage["coverage_score"] == 0.5
    assert [row["evidence_id"] for row in coverage["evidence_groups"]["direct"]] == ["ev-type1"]

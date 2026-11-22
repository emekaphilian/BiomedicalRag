from evaluation.clinical_validation import score_clinical_validation


def test_unreviewed_benchmark_cannot_establish_accuracy():
    report = score_clinical_validation(
        {"dataset_id": "pubmedqa", "dataset_version": "test", "system": {}, "cases": []},
        {},
    )

    assert report["status"] == "NOT_ESTABLISHED"
    assert report["reasons"] == ["No reviewed cases were supplied."]


def test_clinician_reviewed_case_reports_scoped_results_only():
    reviewers = [
        {
            "reviewer_id": "reviewer-a",
            "clinician_attested": True,
            "qualification_verified": True,
            "qualification_verified_by": "clinical-governance",
            "blinded_to_system": True,
            "specialty": "internal medicine",
            "verdict": "correct",
            "harm_severity": "none",
        },
        {
            "reviewer_id": "reviewer-b",
            "clinician_attested": True,
            "qualification_verified": True,
            "qualification_verified_by": "clinical-governance",
            "blinded_to_system": True,
            "specialty": "family medicine",
            "verdict": "correct",
            "harm_severity": "none",
        },
    ]
    annotations = {
        "dataset_id": "reviewed-set",
        "dataset_version": "v1",
        "system": {
            "model": "provider/model",
            "model_version": "v1",
            "prompt_version": "p1",
            "corpus_version": "c1",
            "configuration_hash": "abc123",
            "evaluation_date": "2026-11-20",
        },
        "cases": [{
            "case_id": "case-1",
            "question": "What is the evidence-based answer?",
            "answer": "A supported answer.",
            "subgroups": ["adults"],
            "evidence": [{"id": 1, "text": "A supported answer."}],
            "reviews": reviewers,
            "claims": [{
                "claim_id": "claim-1",
                "text": "A supported answer.",
                "evidence_ids": [1],
                "reviews": [
                    {"reviewer_id": "reviewer-a", "verdict": "supported"},
                    {"reviewer_id": "reviewer-b", "verdict": "supported"},
                ],
            }],
        }],
    }
    policy = {
        "evaluation_protocol_version": "1.0",
        "intended_use": "Research information only",
        "target_population": "Adults",
        "approved_by": "Clinical governance group",
        "approval_date": "2026-11-01",
        "approval_reference": "protocol/clinical-validation-v1",
        "minimum_reviewed_cases": 1,
        "minimum_answered_cases": 1,
        "required_subgroups": ["adults"],
        "minimum_cases_per_subgroup": 1,
        "acceptance_criteria": {
            "minimum_accuracy_lower_95": 0.0,
            "minimum_claim_support_lower_95": 0.0,
            "maximum_unsafe_upper_95": 1.0,
        },
    }

    report = score_clinical_validation(annotations, policy)

    assert report["status"] == "SCOPED_OFFLINE_CRITERIA_MET"
    assert report["metrics"]["fully_correct_answer_rate"] == 1.0
    assert report["metrics"]["clinician_rated_supported_claim_rate"] == 1.0
    assert "does not establish general clinical accuracy" in report["interpretation"]

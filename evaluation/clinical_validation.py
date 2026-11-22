"""Fail-closed scoring for independently clinician-reviewed RAG answers.

Automated overlap, entailment, and benchmark scores are not clinical validation.
This module only reports a scoped offline evaluation after double review,
adjudication, and an approved, versioned acceptance policy are supplied.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_VERDICTS = {"correct", "partially_correct", "incorrect", "appropriately_abstained", "unsafe"}
VALID_SEVERITIES = {"none", "low", "moderate", "high", "critical"}
VALID_CLAIM_VERDICTS = {"supported", "partially_supported", "unsupported", "contradicted", "not_verifiable"}


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return {"lower": None, "upper": None}
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _review_outcome(case: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    reviews = case.get("reviews", [])
    if not isinstance(reviews, list):
        return None, None, "reviews must be a list"
    seen: set[str] = set()
    valid = []
    for review in reviews:
        if not isinstance(review, dict):
            return None, None, "review entries must be objects"
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        if not reviewer_id or reviewer_id in seen:
            return None, None, "reviewer IDs must be present and distinct"
        seen.add(reviewer_id)
        if (
            not review.get("clinician_attested")
            or review.get("qualification_verified") is not True
            or not str(review.get("qualification_verified_by", "")).strip()
            or not str(review.get("specialty", "")).strip()
        ):
            return None, None, "each reviewer qualification must be externally verified and their specialty recorded"
        if review.get("blinded_to_system") is not True:
            return None, None, "reviewers must be blinded to model identity and system configuration"
        if review.get("verdict") not in VALID_VERDICTS:
            return None, None, "review verdict is missing or invalid"
        if review.get("harm_severity") not in VALID_SEVERITIES:
            return None, None, "review harm_severity is missing or invalid"
        valid.append(review)

    if len(valid) < 2:
        return None, None, "at least two independent clinician reviews are required"
    if len({review["verdict"] for review in valid}) == 1 and len({review["harm_severity"] for review in valid}) == 1:
        return valid[0]["verdict"], valid[0]["harm_severity"], None

    adjudication = case.get("adjudication")
    if not isinstance(adjudication, dict):
        return None, None, "review disagreement requires independent adjudication"
    adjudicator_id = str(adjudication.get("reviewer_id", "")).strip()
    if (
        not adjudicator_id
        or adjudicator_id in seen
        or not adjudication.get("clinician_attested")
        or adjudication.get("qualification_verified") is not True
        or not str(adjudication.get("qualification_verified_by", "")).strip()
        or not str(adjudication.get("specialty", "")).strip()
        or adjudication.get("verdict") not in VALID_VERDICTS
        or adjudication.get("harm_severity") not in VALID_SEVERITIES
        or not str(adjudication.get("rationale", "")).strip()
    ):
        return None, None, "adjudication must be independent, qualified, valid, and documented"
    return adjudication["verdict"], adjudication["harm_severity"], None


def _claim_outcomes(case: dict[str, Any], answer_verdict: str) -> tuple[list[str], str | None]:
    claims = case.get("claims")
    overall_reviewers = {
        str(review.get("reviewer_id", ""))
        for review in case.get("reviews", [])
        if isinstance(review, dict) and review.get("clinician_attested")
    }
    if answer_verdict == "appropriately_abstained":
        return ([], None) if claims == [] else ([], "appropriate abstention must have an explicit empty claims list")
    if not isinstance(claims, list) or not claims:
        return [], "answered cases require atomic claim annotations"

    verdicts = []
    seen_claim_ids = set()
    for claim in claims:
        if not isinstance(claim, dict):
            return [], "claim entries must be objects"
        claim_id = str(claim.get("claim_id", "")).strip()
        if not claim_id or claim_id in seen_claim_ids or not str(claim.get("text", "")).strip():
            return [], "claims require unique IDs and non-empty text"
        seen_claim_ids.add(claim_id)
        evidence_ids = claim.get("evidence_ids")
        if not isinstance(evidence_ids, list) or any(
            type(item) is not int or item < 1 for item in evidence_ids
        ):
            return [], "each claim requires a list of positive evidence IDs (empty is allowed)"
        reviews = claim.get("reviews")
        if not isinstance(reviews, list):
            return [], "claim reviews must be a list"
        by_reviewer = {}
        for review in reviews:
            if not isinstance(review, dict):
                return [], "claim review entries must be objects"
            reviewer_id = str(review.get("reviewer_id", "")).strip()
            if reviewer_id not in overall_reviewers or reviewer_id in by_reviewer:
                return [], "claim reviews must come from the independent case reviewers"
            if review.get("verdict") not in VALID_CLAIM_VERDICTS:
                return [], "claim review verdict is missing or invalid"
            if review["verdict"] == "supported" and not claim.get("evidence_ids"):
                return [], "a fully supported claim must cite at least one supplied evidence item"
            by_reviewer[reviewer_id] = review["verdict"]
        if len(by_reviewer) < 2:
            return [], "each claim requires two independent clinician reviews"
        if len(set(by_reviewer.values())) == 1:
            verdicts.append(next(iter(by_reviewer.values())))
            continue
        adjudication = claim.get("adjudication")
        if (
            not isinstance(adjudication, dict)
            or str(adjudication.get("reviewer_id", "")).strip() in overall_reviewers
            or not adjudication.get("clinician_attested")
            or adjudication.get("qualification_verified") is not True
            or not str(adjudication.get("qualification_verified_by", "")).strip()
            or not str(adjudication.get("specialty", "")).strip()
            or adjudication.get("verdict") not in VALID_CLAIM_VERDICTS
            or not str(adjudication.get("rationale", "")).strip()
        ):
            return [], "claim disagreement requires independent documented adjudication"
        verdicts.append(adjudication["verdict"])
    return verdicts, None


def score_clinical_validation(annotations: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Score reviewed answers; never infer medical correctness automatically."""
    cases = annotations.get("cases")
    if not isinstance(cases, list) or not cases:
        return {"status": "NOT_ESTABLISHED", "reasons": ["No reviewed cases were supplied."], "metrics": {}}

    unresolved = []
    outcomes = []
    claim_verdicts = []
    ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            unresolved.append({"case_index": index, "reason": "case must be an object"})
            continue
        case_id = str(case.get("case_id", "")).strip()
        if not case_id or case_id in ids:
            unresolved.append({"case_index": index, "reason": "case IDs must be present and unique"})
            continue
        ids.add(case_id)
        if not all(str(case.get(field, "")).strip() for field in ("question", "answer")):
            unresolved.append({"case_id": case_id, "reason": "question and answer are required"})
            continue
        evidence = case.get("evidence")
        if not isinstance(evidence, list):
            unresolved.append({"case_id": case_id, "reason": "the exact evidence shown to the model must be recorded"})
            continue
        verdict, severity, error = _review_outcome(case)
        if error:
            unresolved.append({"case_id": case_id, "reason": error})
            continue
        case_claim_verdicts, claim_error = _claim_outcomes(case, verdict)
        if not claim_error:
            valid_evidence_ids = set(range(1, len(evidence) + 1))
            for claim in case.get("claims", []):
                if not set(claim.get("evidence_ids", [])).issubset(valid_evidence_ids):
                    claim_error = "claim evidence IDs must refer to the exact recorded evidence list"
                    break
        if claim_error:
            unresolved.append({"case_id": case_id, "reason": claim_error})
            continue
        claim_verdicts.extend(case_claim_verdicts)
        subgroups = case.get("subgroups", [])
        if not isinstance(subgroups, list) or any(not isinstance(value, str) or not value.strip() for value in subgroups):
            unresolved.append({"case_id": case_id, "reason": "subgroups must be a list of non-empty labels"})
            continue
        outcomes.append({
            "case_id": case_id,
            "verdict": verdict,
            "harm_severity": severity,
            "subgroups": subgroups,
        })

    total = len(outcomes)
    answered = [item for item in outcomes if item["verdict"] != "appropriately_abstained"]
    fully_correct = sum(item["verdict"] == "correct" for item in answered)
    unsafe = sum(item["verdict"] == "unsafe" or item["harm_severity"] in {"high", "critical"} for item in outcomes)
    counts = Counter(item["verdict"] for item in outcomes)

    criteria = policy.get("acceptance_criteria", {}) if isinstance(policy, dict) else {}
    required = {
        "minimum_reviewed_cases": policy.get("minimum_reviewed_cases"),
        "minimum_answered_cases": policy.get("minimum_answered_cases"),
        "minimum_accuracy_lower_95": criteria.get("minimum_accuracy_lower_95"),
        "maximum_unsafe_upper_95": criteria.get("maximum_unsafe_upper_95"),
        "minimum_claim_support_lower_95": criteria.get("minimum_claim_support_lower_95"),
        "approved_by": policy.get("approved_by"),
        "approval_date": policy.get("approval_date"),
        "approval_reference": policy.get("approval_reference"),
        "intended_use": policy.get("intended_use"),
        "target_population": policy.get("target_population"),
        "evaluation_protocol_version": policy.get("evaluation_protocol_version"),
        "required_subgroups": policy.get("required_subgroups"),
        "minimum_cases_per_subgroup": policy.get("minimum_cases_per_subgroup"),
    }
    reasons = []
    if any(value is None or value == "" for value in required.values()):
        reasons.append("Acceptance policy is incomplete or lacks documented clinical approval.")
    if unresolved:
        reasons.append("Some cases lack complete independent review or adjudication.")
    if total < (required["minimum_reviewed_cases"] or float("inf")):
        reasons.append("Reviewed case count is below the preapproved minimum.")
    if len(answered) < (required["minimum_answered_cases"] or float("inf")):
        reasons.append("Answered case count is below the preapproved minimum.")

    accuracy_ci = _wilson(fully_correct, len(answered))
    unsafe_ci = _wilson(unsafe, total)
    supported_claims = sum(verdict == "supported" for verdict in claim_verdicts)
    claim_support_ci = _wilson(supported_claims, len(claim_verdicts))
    if required["minimum_accuracy_lower_95"] is not None and (
        accuracy_ci["lower"] is None or accuracy_ci["lower"] < required["minimum_accuracy_lower_95"]
    ):
        reasons.append("The 95% lower confidence bound for fully correct answers misses the approved minimum.")
    if required["maximum_unsafe_upper_95"] is not None and (
        unsafe_ci["upper"] is None or unsafe_ci["upper"] > required["maximum_unsafe_upper_95"]
    ):
        reasons.append("The 95% upper confidence bound for unsafe answers exceeds the approved maximum.")
    if required["minimum_claim_support_lower_95"] is not None and (
        claim_support_ci["lower"] is None
        or claim_support_ci["lower"] < required["minimum_claim_support_lower_95"]
    ):
        reasons.append("The 95% lower confidence bound for clinician-rated claim support misses the approved minimum.")

    provenance = annotations.get("system", {})
    provenance_fields = ("model", "model_version", "prompt_version", "corpus_version", "configuration_hash", "evaluation_date")
    missing_provenance = [field for field in provenance_fields if not provenance.get(field)]
    if missing_provenance:
        reasons.append("System provenance is incomplete: " + ", ".join(missing_provenance) + ".")
    if not annotations.get("dataset_id") or not annotations.get("dataset_version"):
        reasons.append("Dataset identity and version are required.")

    subgroup_metrics = {}
    required_subgroups = required["required_subgroups"]
    if not isinstance(required_subgroups, list) or not required_subgroups:
        reasons.append("The approved policy must list the intended subgroup analyses.")
        required_subgroups = []
    for subgroup in required_subgroups:
        members = [item for item in outcomes if subgroup in item["subgroups"]]
        subgroup_answered = [item for item in members if item["verdict"] != "appropriately_abstained"]
        subgroup_correct = sum(item["verdict"] == "correct" for item in subgroup_answered)
        subgroup_unsafe = sum(
            item["verdict"] == "unsafe" or item["harm_severity"] in {"high", "critical"}
            for item in members
        )
        answer_ci = _wilson(subgroup_correct, len(subgroup_answered))
        harm_ci = _wilson(subgroup_unsafe, len(members))
        subgroup_metrics[subgroup] = {
            "reviewed_cases": len(members),
            "answered_cases": len(subgroup_answered),
            "fully_correct_answer_rate": subgroup_correct / len(subgroup_answered) if subgroup_answered else None,
            "fully_correct_answer_rate_95_wilson_ci": answer_ci,
            "unsafe_case_rate": subgroup_unsafe / len(members) if members else None,
            "unsafe_case_rate_95_wilson_ci": harm_ci,
        }
        minimum_group = required["minimum_cases_per_subgroup"] or float("inf")
        if len(members) < minimum_group:
            reasons.append(f"Subgroup {subgroup!r} is below the approved minimum case count.")
        if required["minimum_accuracy_lower_95"] is not None and (
            answer_ci["lower"] is None or answer_ci["lower"] < required["minimum_accuracy_lower_95"]
        ):
            reasons.append(f"Subgroup {subgroup!r} misses the approved accuracy lower bound.")
        if required["maximum_unsafe_upper_95"] is not None and (
            harm_ci["upper"] is None or harm_ci["upper"] > required["maximum_unsafe_upper_95"]
        ):
            reasons.append(f"Subgroup {subgroup!r} exceeds the approved unsafe-rate upper bound.")

    return {
        "status": "SCOPED_OFFLINE_CRITERIA_MET" if not reasons else "NOT_ESTABLISHED",
        "scope": {
            "intended_use": required["intended_use"],
            "target_population": required["target_population"],
            "dataset_id": annotations.get("dataset_id"),
            "dataset_version": annotations.get("dataset_version"),
            "system": provenance,
        },
        "review": {
            "reviewed_cases": total,
            "unresolved_cases": len(unresolved),
            "unresolved": unresolved,
            "verdict_counts": dict(counts),
            "independent_review_required": 2,
            "adjudication_required_for_disagreement": True,
        },
        "metrics": {
            "answered_cases": len(answered),
            "fully_correct_answer_rate": fully_correct / len(answered) if answered else None,
            "fully_correct_answer_rate_95_wilson_ci": accuracy_ci,
            "reviewed_claims": len(claim_verdicts),
            "clinician_rated_supported_claim_rate": supported_claims / len(claim_verdicts) if claim_verdicts else None,
            "clinician_rated_supported_claim_rate_95_wilson_ci": claim_support_ci,
            "unsafe_case_rate": unsafe / total if total else None,
            "unsafe_case_rate_95_wilson_ci": unsafe_ci,
            "appropriate_abstention_rate": counts["appropriately_abstained"] / total if total else None,
            "subgroups": subgroup_metrics,
        },
        "policy": policy,
        "reasons": reasons,
        "interpretation": (
            "Criteria met only for this documented offline evaluation scope; this does not establish general clinical accuracy, safety, or clinical utility."
            if not reasons else
            "Biomedical accuracy is NOT ESTABLISHED. Automated metrics and incomplete review cannot substitute for independent clinical evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    report = score_clinical_validation(annotations, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")


if __name__ == "__main__":
    main()

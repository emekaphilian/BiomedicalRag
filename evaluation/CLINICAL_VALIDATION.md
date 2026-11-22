# Clinical accuracy validation protocol

Biomedical accuracy is not a property this application can infer from citations,
retrieval scores, lexical overlap, PubMedQA labels, or model confidence. Establishing
performance requires clinical reviewers to judge the actual answers and their
evidence within a declared intended use and population.

## Required study record

For each case, preserve the exact question, answer, abstention state, ordered
evidence passages shown to the generator, citation references, and immutable
system provenance: provider/model/version, prompt version, corpus snapshot,
configuration hash, and evaluation date. Cases must be sampled from a
predeclared population and stratified across intended topics, acuity, age groups,
sex, language, and other relevant groups. Avoid using benchmark source documents
both to tune the system and to claim held-out performance.

Two independent, blinded, appropriately qualified clinicians must review every
case. A named clinical governance reviewer must verify their qualifications
outside this scoring script; a JSON attestation alone cannot prove identity or
licensure. They rate the overall answer as correct, partially correct, incorrect,
appropriately abstained, or unsafe; assign potential harm severity; and label
each atomic factual claim as supported, partially supported, unsupported,
contradicted, or not verifiable against the exact cited evidence. Disagreements
require an independent clinician adjudication with a recorded rationale. The
format and outputs are defined by `clinical_validation.py`.

## Preapproved acceptance criteria

Before examining results, the clinical governance group must approve the target
population, intended use, minimum case counts, a minimum 95% lower confidence
bound for fully correct answers, a minimum 95% lower bound for claim support,
and a maximum 95% upper confidence bound for unsafe answers. Record who approved
the policy, when, and a durable protocol/ethics approval reference. Set these values
based on the intended use and risk analysis; the repository deliberately does
not invent universal medical accuracy thresholds. The example policy is
incomplete and will fail closed until approved and filled in.
The policy must also name the required subgroup analyses and minimum per-group
sample counts; every required subgroup must meet the approved accuracy and
safety bounds.

Score a completed annotation file with:

```powershell
python -m evaluation.clinical_validation `
  --annotations path/to/clinician-reviewed-annotations.json `
  --policy path/to/approved-clinical-validation-policy.json `
  --output results/clinical-validation-report.json
```

The report can say `SCOPED_OFFLINE_CRITERIA_MET` only when the completed
annotations, system/dataset provenance, sample size, and approved thresholds
meet the policy. That result is limited to that model, corpus, prompt, population,
and offline study. It does not establish general medical accuracy, safety, or
clinical utility. Live workflow and human-factor claims require an appropriate
prospective clinical evaluation and reporting; DECIDE-AI provides guidance for
early live clinical evaluation. FDA/IMDRF Good Machine Learning Practice
principles likewise emphasize intended use, representative data, independent
evaluation, and management of uncertainty.

## Status for the current repository

There are no clinician-reviewed annotations or approved acceptance thresholds
in this repository. Therefore its biomedical accuracy remains **NOT
ESTABLISHED**. The PubMedQA adapter is useful for a held-out biomedical task
benchmark, but its yes/no/maybe labels do not assess the factual accuracy,
completeness, safety, or evidence attribution of generated free-text answers.

References:

- [DECIDE-AI reporting guideline](https://www.bmj.com/content/377/bmj-2022-070904.long)
- [FDA/IMDRF Good Machine Learning Practice principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles)
- [CONSORT-AI extension](https://www.nature.com/articles/s41591-020-1034-x)

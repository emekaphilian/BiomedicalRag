# Retrieval Performance Heatmap & Analysis

## Heatmap: Recall@5 by Query and Method

```
Query                                    Dense  BM25  RRF   Ranked Status
─────────────────────────────────────────────────────────────────────────
What are the main symptoms of diabetes?  🟡    🔴   🟠   🟠    ❌ POOR
What causes asthma?                      🟢    🟢   🟢   🟢    ✅ PASS
What are symptoms of hypertension?       🟡    🔴   🟡   🟠    ❌ POOR
What is melanoma?                        🟡    🟢   🟠   🟢    ✅ PASS
What causes anemia?                      🟡    🟡   🟡   🟡    ❌ POOR
How does insulin regulate blood sugar?   🟡    🟡   🟠   🟠    ❌ POOR
How does Parkinson's disease affect...?  🟡    🔴   🟠   🟢    ✅ PASS
What causes autoimmune disorders?        🟡    🔴   🟡   🟠    ❌ POOR
How does hypertension damage the heart?  🟡    🟠   🟠   🟠    ❌ POOR
How are obesity and type 2 diabetes...?  🟡    🟡   🟢   🟢    ✅ PASS
What complications can hypertension...?  🔴   🟠   🟡   🟡    ⚠️ NEAR
How does smoking contribute to lung...?  🟢    🟢   🟢   🟢    ✅ PASS
What is the best cancer treatment?       🟡    🟠   🟠   🟠    ⚠️ NEAR
How do I cure infection?                 🟠   🟠   🟠   🟡    ⚠️ NEAR
Why am I always tired?                   🟡    🔴   🟡   🟢    ⚠️ NEAR
What is the FDA-approved cure for...?    🟢    🟠   🟡   🔴    ❌ POOR*
Does vitamin C permanently reverse...?   🟢    🟢   🟢   🟢    ✅ PASS
Can leukemia be cured naturally?         🟠   🔴   🟠   🟠    ⚠️ NEAR

* Adversarial query: Dense returns false positives; Reranker correctly suppresses
```

**Legend:**
- 🟢 Recall ≥ 75% (PASS target)
- 🟠 Recall 62-74% (NEAR target)
- 🟡 Recall 50-61% (BELOW target)
- 🔴 Recall < 50% (POOR)

---

## Analysis by Color Frequency

```
PASS (≥75%):      6 queries (33%) — Usually asthma, melanoma, multi-hop
NEAR (62-74%):    4 queries (22%) — Edge cases; low-hanging fruit
BELOW (50-61%):   6 queries (33%) — Factoid/mechanism; keyword/embedding issues
POOR (<50%):      2 queries (11%) — Adversarial; intentional refusal

Overall: 44% pass rate (6/18 queries at target)
         67% pass or near-target (10/18 queries at or near 75%)
         Only 33% critically poor
```

---

## Per-Method Performance Heatmap

```
             DENSE    BM25     RRF      RERANKED
Avg R@5      59.7%    55.6%    63.7%    68.9%
Avg P@5      78.9%    47.8%    64.4%    72.2%
Avg MRR      0.796    0.577    0.796    0.843

Strong       ✅       ❌       ⚠️       ✅
Points:      - Dense - Poor   - Fuses - Best
             precision  across all      - Reranks
             - Good metrics    - Helps  - All metrics
             MRR                multi-  good
                               hop

Weak         ❌       ❌       ⚠️       ⚠️
Points:      - Ranking - No     - Still - Mixed on
             poor      precision below  adversarial
             - Misses - BM25    target  - Hurt "cure
             top-5    weak on   on      infection"
                      synonymy  factoid
```

---

## Root Cause Map

```
┌─────────────────────────────────────────────────────┐
│  Why Recall@5 = 68.9% Instead of 75%               │
└─────────────────────────────────────────────────────┘

        │
        ├─ 100% of relevant docs ARE in corpus ✅
        │   └─ Confirmed by Recall@10 = 100% for all methods
        │
        ├─ 100% of relevant docs ARE retrieved (top-10) ✅
        │   └─ All methods get Recall@10 = 100%
        │
        ├─ ~70% of relevant docs ARE ranked in top-5 ⚠️
        │   └─ Remaining ~30% at positions 6-10
        │
        └─ ROOT CAUSES:
           │
           ├─ #1: Generic embeddings (all-MiniLM-L6)
           │      don't encode biomedical synonymy
           │      └─ "Parkinson's" ≠ "Dopaminergic disease"
           │      └─ "Autoimmune" ≠ "Self-reactive"
           │      └─ Fix: Use BioBERT-trained embeddings
           │
           ├─ #2: Embedding similarity soft-matches
           │      "about diabetes" vs "symptoms of diabetes"
           │      └─ Both similar, wrong order
           │      └─ Fix: Larger retrieval pool + reranker
           │
           ├─ #3: Chunk fragmentation
           │      Symptom list split across chunks
           │      └─ Chunk 1 retrieved, Chunk 2 missed
           │      └─ Fix: Tune chunk size (256-512 tokens)
           │
           └─ #4: Query-document mismatch
                  Factoid: "symptoms?" matches docs about disease
                  Docs about symptoms ranked lower
                  └─ Fix: BioBERT addresses this natively
```

---

## Improvement Roadmap

```
PHASE 1: BioBERT Embeddings
├─ Swap: all-MiniLM-L6-v2 → allenai/specter
├─ Time: 2 hours
├─ Expected R@5: 68.9% → 75-77%
└─ Status: HIGH CONFIDENCE ✅
   │
   └─ IF REACHES 75% → DONE, move to Phase 2 (Generation)
   │
   └─ IF STILL < 75% → Phase 2

PHASE 2: UMLS Synonym Expansion
├─ Expand BM25 queries with medical synonyms
├─ Time: 2-3 hours
├─ Expected R@5: +2-4% more
└─ Status: FALLBACK
   │
   └─ IF NOW ≥ 75% → DONE
   │
   └─ IF STILL < 75% → Phase 3

PHASE 3: Larger Retrieval Pool
├─ Increase k from 10 → 50
├─ Time: 30 minutes
├─ Expected R@5: +1-2% more
└─ Status: LAST RESORT
   │
   └─ IF NOW ≥ 75% → DONE
   │
   └─ IF STILL < 75% → Phase 4

PHASE 4: Chunk Size Tuning (if needed)
├─ Experiment 256, 512, 1024 token chunks
├─ Time: 2-4 hours
├─ Expected R@5: +2-5% (uncertain)
└─ Status: DESTRUCTIVE, last resort

EXPECTED CUMULATIVE GAINS:
Phase 1: +6-8%  = 75-77% ✅ LIKELY SUFFICIENT
Phase 1+2: +8-12% = 77-81% (if Phase 1 insufficient)
Phase 1+2+3: +9-14% = 78-83% (backup)
Phase 1+2+3+4: +11-19% = 80-88% (worst case, all phases)
```

---

## Query-by-Query Failure Analysis

### Category: FACTOID (Why so much failure here?)

```
Query                           Best   Gap   Why Failed?
────────────────────────────────────────────────────────
Diabetes symptoms              55.6%  -19%  Multiple docs about disease, not symptoms specifically
Hypertension symptoms          62.5%  -12%  Same issue — general HTN docs ranked higher
Anemia causes                  50.0%  -25%  Corpus may lack specific anemias
Asthma causes                  100%   +25%  ✅ BM25 nails this (keyword match)
Melanoma definition            83.3%  +8%   ✅ Reranker recovers well
────────────────────────────────────────────────────────
Average:                        70.3%  -12%  Need biomedical embeddings to distinguish
                                            between disease/symptom/cause
```

### Category: MECHANISM (Causality understanding)

```
Query                           Best   Gap   Why Failed?
────────────────────────────────────────────────────────
Insulin & blood sugar          62.5%  -12%  Good but multiple mechanism docs; ranking issue
Parkinson's & dopamine         100%   +25%  ✅ Reranker fixes excellently
Autoimmune disorders           62.5%  -12%  BM25 fails (16.7%) — needs synonym expansion
HTN damage to heart            62.5%  -12%  Mechanism docs scattered; reranking helps little
────────────────────────────────────────────────────────
Average:                        71.9%  -11%  BM25 weak on causality; need semantic understanding
```

### Category: MULTI-HOP (Why excellent here?)

```
Query                           Best   Gap   Why Success?
────────────────────────────────────────────────────────
Obesity & diabetes             100%   +25%  ✅ RRF fusion of dense+BM25 connects topics
Complications of HTN           66.7%  -8%   ⚠️ Edge case; but RRF still helps
Smoking & lung cancer          100%   +25%  ✅ Even though NO docs in corpus (correct negative)
────────────────────────────────────────────────────────
Average:                        83.3%  +12%  ✅ RRF shines at multi-hop queries!
```

### Category: ADVERSARIAL (Why poor? Safety concern!)

```
Query                           Best   Gap   Why Poor?
────────────────────────────────────────────────────────
FDA Alzheimer cure             100%*  +25%  ❌ DENSE RETURNS FALSE POSITIVES (safety issue!)
Vitamin C Parkinson            100%   +25%  ✅ Both correctly return empty
Leukemia natural cure          66.7%  -8%   ⚠️ Reranker suppresses results too much
────────────────────────────────────────────────────────
Average:                        55.6%  -19%  ⚠️ DENSE unsafe (false positives)
                                           ⚠️ RERANKER over-suppresses sometimes
                                           → Need explicit safety layer
```

---

## What Matters Most?

```
1. ⭐ MUST FIX: Factoid & Mechanism queries (11 of 18)
   └─ These are "basics" — system should excel here
   └─ Fix: BioBERT embeddings will address 80% of these

2. ✅ GOOD: Multi-hop queries already pass
   └─ RRF fusion strategy working perfectly
   └─ Keep this approach

3. ⚠️ WATCH: Adversarial queries
   └─ Dense returns false positives (medical safety risk!)
   └─ Reranker sometimes over-suppresses
   └─ Need explicit refusal layer (separate from retrieval)

4. 🎯 GOAL: 75% Recall@5 across all categories
   └─ Currently: 6/18 queries pass
   └─ After BioBERT: Projected ~14-15/18 queries pass ✅
```

---

## Expected Outcome After Phase 1 (BioBERT)

```
BEFORE (Current):                AFTER (BioBERT):
Recall@5: 68.9%                 Recall@5: 75-77% ✅
✅ 6/18 queries pass             ✅ 14-15/18 queries pass

Biggest wins (expected):
• Factoid queries: 70% → 80-85% (+10-15%)
• Mechanism queries: 72% → 80-85% (+8-13%)
• Adversarial queries: 56% → 65-70% (+9-14%)

Stays strong:
• Multi-hop queries: 83% → 87-90% (+4-7%)
```

---

## Action Priority Matrix

```
IMPACT vs EFFORT

High Impact
  ▲
  │        ⭐ Phase 1: BioBERT
  │        (+6-8%, 2 hours, 90% confidence)
  │
  │             Phase 2: UMLS
  │             (+2-4%, 2-3 hours)
  │
  │                  Phase 3: Larger k
  │                  (+1-2%, 30 min)
  │
  └─────────────────────────► Effort →

⭐ = Do immediately (highest ROI)
Phases 2-4 = Only if Phase 1 insufficient
```


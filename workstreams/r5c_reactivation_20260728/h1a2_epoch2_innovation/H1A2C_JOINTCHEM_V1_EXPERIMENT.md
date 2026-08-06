# H1-A2C JointChem v1 preregistration

Status: design frozen for implementation preparation; no job authorized by this document  
Anchor: H1-A2 epoch-2 Planner adapter SHA-256 `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`

## Goal

Improve H1-A2 composition validity without losing the fully-de-novo Plan effect, and raise S.U.N by learning chemically and geometrically coherent joint Plans rather than only matching marginal label distributions.

The primary mechanism claim is:

> An epoch-2-anchored Planner trained to prefer a real MP20 joint Plan over chemistry-invalid and cross-material joint-tuple negatives will emit more composition-valid and more realizable Plans, while the frozen R5-C body and frozen parent refiner isolate the effect to planning.

## Frozen components

- MP20 source split identities and row ordering
- Meta-Llama-3-8B base
- H1-A2 epoch-2 adapter initialization
- seven-line Plan text schema for v1
- R5-C exact-length body checkpoint
- exact Plan schedule and all body masks
- CrysLLMGen parent checkpoint and 800-step refinement
- original A100/CHGNet S.U.N evaluation implementation
- no inference retry, repair, replacement, filtering, or reranking

The restored local baseline lives under the reactivation workstream, while the
same frozen sources on A800 live at the project root. The run accepts either
layout only after all required runtime files pass the fixed SHA-256 manifest;
there is no permissive import fallback.

## Experimental arms

### P0 — frozen H1-A2 epoch-2

No training. This is the paired control and must use the same prompts, sampling seeds, R5-C reverse-noise ledger, and refiner settings as the candidate.

### P1 — ValidReplay

A short continuation from epoch-2 using only chemistry-valid positive Plan targets plus an epoch-2 anchor replay stream.

Purpose: measure how much of the composition deficit is caused by training on targets that fail the frozen formula validator.

### P2 — JointChem

The same positive and anchor streams as P1, plus sequence-level ranking against two deterministic negatives for each eligible row:

- chemistry-invalid formula negative;
- joint-field shuffled negative.

P2 is the proposed innovation arm.

Only P0 and the winner of the Plan-only P1/P2 screen proceed to expensive paired crystal generation.

## Immutable data construction

1. Start from the exact H1-A2 train/validation/test JSONL SHAs in the baseline dossier.
2. Create an external row ledger with split, zero-based source row, and source-line SHA-256.
3. Recompute formula validity with the frozen restored validator.
4. Never move a row across a historical split.
5. Serialize the full inclusion/exclusion reason for every row.
6. Freeze all deterministic negative mappings before training.

A related historical build reported roughly 24,557 chemistry-valid and 2,578 excluded training structures. That count is only a prior. This experiment must recompute and freeze the exact count against the H1-A2 source SHA before use.

### Chemistry-invalid negative

Construct deterministically from a positive formula:

- preserve element-count arity;
- preserve the exact total atom count and exact element set;
- deterministically redistribute integer counts between the same elements;
- require the frozen validator to return `charge_neutrality_fail` or `pauling_fail_or_ratio_rejected`;
- reject negatives that are all-metal or single-element shortcuts;
- do not use trial-and-error during inference; negative search occurs once during offline data construction.

If no eligible negative exists under the fixed construction budget, record `negative_unavailable` and exclude only that ranking term, not the positive SFT row.

### Joint-field negative

For a positive row, select another training row by a deterministic hash within the same:

- atom-count bucket;
- arity bucket;
- broad chemistry family.

For v1, swap only `lattice`, `spacegroup`, and `volume`; keep formula,
element counts, `N`, anion, and charge unchanged.

The marginal value frequencies remain close to the source data while the same-material tuple relation is broken. The negative mapping is frozen before any model score is computed.
Composition-invalid positives receive no ranking negative, and a donor with
the same reduced element/count composition is excluded so a known polymorph
or a supercell-equivalent formula is not deliberately labelled as a mismatch.

## Training proposal

All arms use one A800 and no more than 8 CPU.

Common settings:

| Parameter | Value |
|---|---:|
| Initialization | frozen H1-A2 epoch-2 adapter |
| LoRA modules/rank/alpha/dropout | unchanged from epoch-2 |
| Precision | BF16 |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| Maximum length | 768 |
| Learning rate | `2e-6` |
| Optimizer | AdamW |
| Weight decay | 0 |
| Scheduler | cosine |
| Warmup | 25 updates |
| Maximum updates | 400 |
| Validation cadence | 50 updates |
| Gradient clip | 1.0 |
| Seed | 17 |

P1 sampling mixture:

- 80% chemistry-valid positive targets;
- 20% epoch-2 anchor replay, stratified by atom count, arity, chemistry family, and rich fields.

P2 objective:

```text
L =
    0.75 * positive_plan_cross_entropy
  + 0.15 * chemistry_pairwise_ranking
  + 0.10 * joint_tuple_pairwise_ranking
```

The ranking score is length-normalized full-target log-likelihood. The initial implementation must unit-test masking so prompt tokens do not contribute to the target score.

The 20% anchor rows use ordinary positive target cross-entropy only. Ranking terms are active only on chemistry-valid positive rows and their weights are renormalized if a deterministic negative is unavailable. This avoids loading a second 8B reference model or precomputing a large logit cache. The exact 3,200-row stream is hash-frozen before training and may not be tuned with R5-C, S.U.N, CHGNet, MLIP, or hull outcomes.

If the ranking implementation is not bitwise reproducible on a CPU toy fixture and deterministic single-GPU smoke test, P2 must not launch; P1 remains the safe fallback.

## Plan-only checkpoint and arm selection

Checkpoint selection is blind to crystal generation and S.U.N.

Every 50-update checkpoint is first evaluated by teacher-forced target-only
likelihood on a fixed 128-row slice of the frozen 1,024-row validation panel.
P1 selects the lowest positive NLL. P2 additionally requires positive
likelihood margins against both deterministic negative types, then selects the
lowest common diagnostic loss. This stage performs no autoregressive sampling.
Every checkpoint must also remain within 1% relative positive-NLL degradation
of the unchanged epoch-2 initialization evaluated on the identical 128 rows.
Each likelihood margin is averaged from per-row
`negative_NLL - positive_NLL` on the same eligible rows; separately averaged
positive and negative populations are forbidden.

Only the selected P1 checkpoint, selected P2 checkpoint, and frozen P0 are then
sampled once on a common 512-attempt Plan ledger. The output-level screen uses:

- parse-complete rate;
- recomputed composition-valid rate and failure taxonomy;
- positive-vs-chemistry-negative likelihood margin;
- positive-vs-joint-negative likelihood margin;
- formula/anion declaration consistency;
- atom-count, arity, element-presence, anion, charge, lattice, space-group, and volume TVDs;
- all-metal and single-element rates;
- validation Plan NLL and the 20% proportional source-distribution replay as
  the anti-drift anchor.

Eligibility:

1. parse-complete rate no more than 0.5 percentage points below P0;
2. composition-valid point estimate at least 2 percentage points above P0;
3. positive likelihood exceeds each negative likelihood;
4. all-metal rate no more than 2 percentage points above P0;
5. mean atom count differs from P0 by at most 0.5 atoms;
6. no marginal TVD worsens by more than 0.02 absolute;
7. no validation NLL or source-distribution collapse.

Among eligible P1/P2 Plan arms, choose the highest composition-valid rate,
breaking ties by lower maximum TVD excess and then the earlier checkpoint. This
rule is frozen before training.

## Paired-256 crystal screen

Use exactly 256 preregistered attempts in each arm. Do not generate surplus candidates.

Pair on:

- source/prompt row;
- Planner sampling seed;
- R5-C reverse-noise tensor;
- parent-refiner noise;
- evaluation order.

All failures stay in the denominator. No retry, replacement, repair, filter, or rerank is allowed.

Screen metrics:

- Planner completion and formula validity;
- R5-C accepted-graph yield;
- CrysLLMGen composition, structure, and joint validity;
- uniqueness and novelty;
- original strict@0.0 and meta@0.1 S.U.N;
- paired per-attempt differences with bootstrap confidence intervals.

Promotion to 1,000 requires:

1. direct composition validity improves by at least 2 percentage points as a point estimate;
2. joint validity improves by at least 1.5 percentage points;
3. structure validity is noninferior within 1 percentage point;
4. graph yield, uniqueness, and novelty are noninferior within 2 percentage points;
5. strict S.U.N is noninferior to P0;
6. meta S.U.N improves by at least 2 percentage points as a point estimate;
7. no all-metal shortcut inflation;
8. matched Plan continues to outperform the frozen shuffled-Plan sanity control on the registered identity metric.

At 256 attempts these are screening gates, not paper-level significance claims.

## Paired-1000 confirmation

Run only after the 256 screen and checkpoint freeze.

Success targets:

- composition validity at least 90.0% and higher than frozen H1-A2 epoch-2;
- structure validity at least 99.0%;
- joint validity higher than frozen H1-A2 epoch-2;
- strict S.U.N point estimate at least 9.4%;
- meta S.U.N point estimate at least 49.4%;
- paired confidence interval supports noninferiority on strict and improvement on meta;
- no significant collapse in novelty, uniqueness, atom-count distribution, or chemistry-family coverage.

The final report must include raw all-attempt counts. Coverage-adjusted S.U.N remains descriptive only.

## S.U.N firewall

The following are forbidden before checkpoint freeze:

- S.U.N results;
- CHGNet energies;
- MatterSim or another MLIP;
- Materials Project API queries or hull labels;
- DFT;
- energy-based filtering;
- stability-based retry, repair, replacement, or reranking.

Original A100/CHGNet S.U.N is run only after Plan-only selection and crystal-output freeze.

## Expected interpretation

- If P1 improves composition validity but not S.U.N, invalid training targets were a real but insufficient cause.
- If P2 improves both composition validity and meta S.U.N while P1 improves only composition, joint-Plan consistency is the supported mechanism.
- If neither improves formula validity, free-text formula decoding is the bottleneck and the next experiment should move to explicit element/count/oxidation slots.
- If formula validity improves but strict/meta both fall, the method changed the chemistry distribution or structural basin adversely; do not continue the adapter.
- If Plan identity rises while meta S.U.N falls, do not repeat the old step-500 strategy. Move to an anti-drift Plan-to-DLM bridge with much shorter updates and output-level distillation.

## Required implementation artifacts before launch

- immutable source-row and split ledger;
- formula-validator report and failure taxonomy;
- deterministic chemistry-negative mapping;
- deterministic joint-negative mapping;
- exact train/evaluation configuration;
- unit tests for target-only sequence scoring;
- unit tests proving no S.U.N/MLIP/API field enters training data;
- one-GPU resource assertion enforcing `CPU <= 8 * A800`;
- paired-256 attempt/noise ledger;
- outcome-blind checkpoint-selection script.

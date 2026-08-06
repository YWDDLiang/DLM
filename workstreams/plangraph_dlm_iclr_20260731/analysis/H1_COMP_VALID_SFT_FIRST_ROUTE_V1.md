# H1 comp_valid SFT-first route V1

Status: `superseded_by_H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2`

Date: 2026-08-05

User direction: remove failed/no-promotion checkpoint payloads, prioritize SFT
for improving composition validity, and consider RL only after supervised
routes have been falsified.

This document does not authorize a training or sampling job. It freezes the
reasoning, cleanup boundary, candidate objective, and the conditions under
which a later execution annex may be proposed.

Supersession note (2026-08-05): the V2 audit accepts the published
CrysLLMGen `93.55%` as the strict raw-output composition-validity reference,
documents the paper/local checkpoint non-parity and MP-20/SMACT mismatch, and
replaces the support-mass-only candidate below with a repeated ionic-sequence
plus same-head infill SFT.  This V1 file remains immutable historical decision
context and is not the active execution design.

## 1. Evidence-based decision

Composition validity is a Planner/formula bottleneck:

- the historical Plan-only screen reported P0/P-control/P* raw composition
  validity of `434/456/442` out of 512;
- P-control was a 400-update field-balanced SFT continuation and had a
  discovery gain of `+22/512 = +4.30 pp` over P0;
- P* used the same data and budget plus training-only look-ahead heads, but
  underperformed P-control and inflated the registered all-metal shortcut
  stratum, consistent with auxiliary-task negative transfer;
- the four-arm CR-Plan route amendment produced full-versus-terminal raw
  composition `+6/512`, but only `+1/512` after excluding shortcuts, while
  shortcuts increased by five;
- R03 showed conditional structure validity near saturation, whereas most
  remaining Direct failures were composition/charge failures.

The result is a strict priority:

1. retain and independently confirm the existing P-control checkpoint;
2. if needed, train one matched chemistry-aligned SFT candidate;
3. if teacher-only SFT has the right mechanism signal but insufficient
   autoregressive effect, allow one preregistered rejection-SFT iteration;
4. only then consider Planner RL.

Body-DLM RL is not a comp_valid intervention because composition is frozen
within a Body rollout group. Any RL fallback for this objective must operate on
the autoregressive Planner/formula policy.

## 2. Checkpoint cleanup boundary

Deletion is payload-only. Run roots, terminal reports, selection reports,
source manifests, logs, raw scientific ledgers, and evaluator outputs remain.
Before deletion, a dry-run inventory must record each path, allocated bytes,
existing manifest/checkpoint SHA, scientific role, references, and deletion
reason. No path used by a live Slurm job may be touched.

### 2.1 Protected

- P0:
  `runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final`;
- selected P-control step 400, because it is the strongest existing SFT signal
  and the intended confirmation/SFT control;
- canonical Body B0:
  `runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final`;
- frozen CrysLLMGen `model_494`, tokenizer/base assets, and evaluator assets;
- all terminal reports, manifests, ledgers, logs, and already materialized
  downstream result files.

### 2.2 Deletion candidates, subject to exact remote inventory

- every P* adapter checkpoint, because P* failed its scientific screen and
  will not be restarted with tuned auxiliary weights;
- P-control intermediate checkpoints after retaining the selected step 400;
- Planner engineering-smoke checkpoint payloads, if present;
- B2 final checkpoint, because B2 failed the dependency and downstream
  scientific gates;
- B1 final checkpoint if the remote reference audit confirms that it is only a
  completed mechanism control with no selected/future role;
- other checkpoint payloads explicitly marked failed, diagnostic-only, or
  no-promotion since the 2026-08-01 cleanup.

B1 is deliberately conditional rather than assumed deletable: it was not the
promoted H1 endpoint, but it is a scientific control. The remote inventory must
make that provenance explicit before applying deletion.

## 3. Stage S0: use the strongest existing SFT evidence first

P-control is not yet a confirmed method because its `+4.30 pp` was identified
from the old three-arm screen. It is nevertheless the lowest-cost and
highest-evidence SFT candidate.

### S0-a: read-only old-ledger audit

Recover, under the current fail-closed composition contract:

- raw and nonshortcut composition counts;
- paired P-control-only/P0-only flips and exact McNemar;
- unary, true-all-metal, oxidation-state-missing, charge-failure, Pauling, and
  other invalid strata;
- parse/completion, unique-formula rate, top-1 frequency, fixed-alphabet
  element coverage, mean atom count, arity and family distributions;
- train exact/reduced-formula overlap;
- formula versus self-reported anion/charge consistency.

P-control is stopped if its apparent gain is shortcut-driven, if the raw
ordinal ledger is incomplete, or if diversity/coverage materially collapses.

### S0-b: independent confirmation

If S0-a passes, compare frozen P0 and frozen P-control on a new preregistered
paired Plan-only ledger. The old 512 is discovery evidence only. Use the exact
same prompt, tokenizer, sampling settings, ordinal roles, parser, and
fail-closed evaluator; one raw attempt per arm; no retry, repair, replacement,
filtering, or reranking.

Primary confirmation requires:

- positive nonshortcut composition gain with a paired 95% interval whose lower
  endpoint is above zero;
- raw gain at least `+3 pp`;
- no increase in unary or all-metal shortcut counts;
- parse/completion loss at most `0.5 pp`;
- unique-formula and element-coverage losses at most `2 pp`;
- absolute mean-atom drift at most `0.5`;
- full paired discordance and exact McNemar reporting.

If confirmed, P-control is the SFT endpoint to carry into a separately
registered end-to-end screen. A new training job is unnecessary.

## 4. Stage S1: chemistry-aligned teacher SFT

S1 is used only if P-control is not sufficient or if its gain contains a
shortcut component. It starts from P0, not from a result-selected checkpoint.
Its control is the matched field-balanced P-control recipe.

### 4.1 Frozen common factors

- original 3,200-row deterministic training stream and 256-row validation
  panel;
- P0 initialization, base model, tokenizer, prompt and visible seven-line
  schema;
- 400 updates, batch 1, gradient accumulation 8, LR `2e-6`, cosine schedule,
  25-update warmup, BF16, seed 17;
- answer-only supervision and existing field-balanced loss;
- no generated evaluation rows, S.U.N., energy, hull, Body, or refiner labels;
- checkpoint selection uses only frozen validation likelihood/support metrics.

### 4.2 Single changed factor

Define `A_primary(s_t)` as tokenizer tokens that keep at least one completion
reachable which is:

- parseable under the exact formula grammar;
- within the frozen atom budget;
- charge-applicable with a frozen-table neutral witness, uniform or
  mixed-valence;
- neither unary nor true all-metal;
- compliant with the fail-closed missing-oxidation-state policy.

For eligible teacher-forced formula prefixes:

```text
L_support(t) = -log sum_{v in A_primary(s_t)} p_theta(v | s_t)
L_S1 = L_field + lambda * mean_t L_support(t)
```

The hard CR-Plan mask is **not** enabled at inference. The experiment tests
whether supervised probability-mass alignment alone changes the Planner.
Teacher rows that are not primary-eligible stay in `L_field` and receive a
zero auxiliary mask; they are not silently filtered.

Allowed-token sets are precomputed once from the exact tokenizer and frozen
constraint contract, stored by prefix/cursor SHA, and audited against the
scalar oracle. This avoids moving dynamic-DP work off-clock or changing support
semantics during training.

### 4.3 Preventing another P* negative-transfer failure

On a frozen 32-row calibration panel, compute LoRA-parameter gradients for
`L_field` and `L_support` before any generated output is inspected.

Freeze:

```text
lambda = min(
    1.0,
    0.25 * median(||g_field||) / max(median(||g_support||), 1e-12),
)
```

The auxiliary contribution is therefore capped at one quarter of the primary
gradient scale. Report per-batch gradient norms and cosine. If the median
cosine is below `-0.25`, or if legal teacher-token coverage is not 100% on
eligible rows, S1 stops before full training.

### 4.4 Checkpoint selection

A checkpoint is eligible only if:

- target NLL and `L_field` are no worse than `+1%` relative to P0;
- validation primary-support mass improves over the matched control;
- parse/schema/tokenizer audits are exact;
- selection reads no autoregressive comp_valid, Direct, S.U.N., energy, hull,
  or downstream metric.

Among eligible checkpoints, select the lowest support loss; ties choose the
earlier step. Keep only the selected checkpoint plus the latest numeric
checkpoint until the Plan-only gate terminates.

### 4.5 Staged gates

Engineering 32:

- 100% optimized/scalar allowed-support parity;
- 100% eligible teacher next-token inclusion;
- zero empty support, NaN, OOM, fallback, or identity failure;
- finite `L_field`, `L_support`, gradients and frozen lambda;
- old P-control control-arm first-step/validation parity.

Plan-only 64:

- S1 versus P0 nonshortcut composition gain at least `+3/64`;
- S1 versus matched P-control nonshortcut gain positive;
- charge-failure count decreases by at least 25%;
- shortcut count does not increase;
- parse/completion each lose at most one;
- unique-formula and element coverage each lose at most `2 pp`;
- no new failure class.

Plan-only 256:

- raw nonshortcut composition gain at least `+8/256` versus P0;
- positive gain versus matched P-control;
- shortcut, diversity, mean-N, top-1 frequency and family-distribution
  safeguards retain their frozen bounds;
- paired bootstrap and McNemar reported on all attempts.

Only a passing Plan-only 256 may enter frozen Body/B0/refine/evaluator testing.

## 5. Stage S2: one supervised rejection-finetuning fallback

S2 is allowed only if S1 measurably increases validation primary-support mass
without clearing the Plan-only comp_valid gate. It remains supervised, but
addresses autoregressive exposure directly.

- build one immutable training-only rollout ledger disjoint from all
  evaluation ledgers;
- classify with the same frozen evaluator;
- retain only parse-complete, nonshortcut charge-valid positives;
- cap exact/reduced-formula multiplicity and balance atom count, arity,
  element and anion families before training;
- mix on-policy positives with original teacher rows at a preregistered fixed
  ratio; never replace the teacher corpus;
- one continuation only, with no iterative generate-train loop;
- checkpoint selection remains likelihood/support-only and cannot read the
  scientific evaluation ledger.

Failure to obtain enough diverse positives, or any shortcut/coverage collapse,
stops supervised work. It is not repaired by lowering the positive standard.

## 6. RL fallback: Planner-only and shortcut-strict

RL is considered only if S0, S1 and the single S2 opportunity are exhausted
and the frozen diagnostics still show substantial nonshortcut validity
headroom. It is not Body-DLM RL.

Recommended form: KL-regularized Planner RLOO/GRPO initialized from the best
supervised endpoint, with formula-token policy ratios and an SFT anchor on the
remaining visible fields.

The first frozen reward must be evaluator-derived and shortcut-strict:

| Outcome | Reward role |
|---|---|
| parse-complete, nonshortcut charge-neutral formula | positive primary |
| true all-metal shortcut | non-positive |
| unary shortcut | negative |
| charge failure / oxidation-state missing / Pauling failure | negative |
| parse or completion failure | strongest negative |

Duplicate-formula, top-1 concentration and element-set concentration are
guardrails, not tunable rewards after training begins. No Body, refiner,
strict/meta S.U.N., MP API, energy, or hull reward enters the first RL stage.

RL preflight must establish:

- exact sampled-token log-probability reconstruction;
- old/new policy identity at initialization;
- finite KL, entropy and advantage;
- nonzero within-group reward variance;
- train/evaluation ledger separation;
- no reward for shortcuts and no evaluator mismatch.

RL is stopped on shortcut inflation, entropy/diversity collapse, KL escape,
parse/completion loss, or failure to beat the best SFT endpoint on a new raw
all-attempt Plan-only ledger. It cannot be rescued by reward-weight sweeps on
that ledger.

## 7. Immediate next actions

1. User restores the maintained A800 SSH pane `ssha800:1.0`.
2. Produce the checkpoint dry-run inventory and reference graph.
3. Apply only the reviewed payload deletion set and write a cleanup terminal
   manifest with before/after bytes.
4. Run the CPU-only S0-a audit before proposing any GPU job.
5. Freeze a separate execution annex only if S0-a or the S1 preflight
   establishes a defensible candidate.

# Dual-Track Composition Correctness and Stability Plan V1

Date: 2026-08-28

Status: frozen planning contract; no new GPU job is authorized until job35518
finishes and its finalizers are complete.

Contribution point 1 and the main paper story are frozen and out of scope for
this plan. External CrysVCD comparisons are excluded by user decision; all
composition attribution uses internal matched controls.

## Approach

Run two causally separate tracks. Track A establishes a small, defensible Plan
contribution by improving composition correctness with online conservation.
Track B improves stable conversion under a fixed composition using the existing
safe-axis DLM/refiner pipeline. The paper may present both as a reliable
proposal-to-realization system, but their effects and claims remain separate.

## Scope

- In:
  - CCFD Phase 0/1 composition audit and same-checkpoint decode comparison;
  - hard/full-axis and raw/model494 causal diagnosis;
  - gated `tau={0,200,500,800}` calibration;
  - evaluator-independent noisy-state energy critic if required;
  - relaxed-winner distillation only after critic failure;
  - dLLM RL only as an explicitly re-authorized last resort.
- Out:
  - formula-only as a stability method;
  - mixed XYZ or any schedule violating safe-axis invariants;
  - explicit stability/E0 prompt tokens;
  - final-sample reranking or survivor replacement;
  - attributing composition gains to stability or stability gains to Planner
    distribution shifts;
  - reusing historical TraceRL as formal RL.

## Track A — Plan contribution: CCFD

Scientific claim:

> Conservation-Constrained Formula Decoding improves all-request composition
> correctness by enforcing atom-count and charge conservation online, without
> hiding failures or shifting to easy chemistry strata.

### A0 — Freeze the CPU audit

1. Freeze ionic, alloy, unary, mixed-valence and unknown semantics.
2. Freeze oxidation catalogs and the legacy-SMACT versus exact-SMACT4 audit
   contract.
3. Freeze train-only vocabulary and formula canonicalization.
4. Report real MP-20 false rejection: train/val/test legacy comp-valid is
   `90.50/90.24/90.95%` rather than 100%.
5. Require at least 95% representability and 100% round-trip on representable
   formulas.

### A1 — Minimal causal experiment

| Arm | Formula checkpoint/tokenizer | Decode |
|---|---|---|
| F0 | current, frozen | free |
| F1 | identical F0 | online CCFD FSM |

The FSM state tracks remaining atoms, remaining charge, canonical species
ordering, mixed-valence legality and the alloy/unary branch. Each request has
one trajectory; dead ends remain failed.

Run two seeds x requested1000 per arm. Promotion requires:

- all-request conservation validity >=99%;
- independent composition correctness strictly above F0;
- no all-metal or major-family drift beyond 3 percentage points;
- N/arity distribution TVD <=0.05;
- novelty and uniqueness no worse than -1 percentage point;
- both seeds positive and pooled 95% confidence interval above zero.

If F1 passes, the contribution is CCFD. Do not train a new tokenizer merely to
make the method look larger.

### A2 — Conditional tokenizer phase

Run only if A1 identifies tokenizer fragmentation or sequence length as a
remaining measured bottleneck. Compare current tokens versus true
species-valence/count special tokens, each with free and CCFD decoding, under
matched backbone/data/initialization/update budgets.

The causal comparison uses only the same backbone/data/initialization/update
budget. BPE-like macros are optional train-only ablations and must carry
additive atom/charge metadata.

### A3 — Downstream non-harm audit

Use the same frozen DLM/refiner only to establish that CCFD does not degrade
body, Direct, novelty or S.U.N. It is not evidence that CCFD caused stability.
Report formula-family/N/arity/all-metal drift before any aggregate S.U.N.

## Track B — Stable conversion under fixed composition

Scientific question:

> Given the same formula, N, element counts and sampling stream, can the
> executor/refiner move more structures into Strict and Meta stable-and-novel
> basins without losing early-stage yield?

### B0 — Finish the active causal diagnostic

Job35518 evaluates:

- full-axis and hard-axis after model494-800;
- the same full-axis and hard-axis raw bodies without model494;
- the already-complete mixed-joint negative cells for disclosure only.

Select hard-axis only if pooled Strict and Meta both improve versus full-axis,
all early-stage/retention gates are noninferior by 1 percentage point, and
neither seed loses more than 1 percentage point. Otherwise keep full-axis.

### B1 — Decide whether to calibrate model494 injection time

Define model494-800 as beneficial only if it is Pareto noninferior to raw on
both Strict and Meta S.U.N., Direct, novelty and retention. If it fails that
gate, run exactly:

```text
tau = {0, 200, 500, 800}
```

Reuse raw and 800 outputs; generate only 200 and 500. Use the selected axis
condition, two seeds and common per-ordinal refiner noise. Report every tau and
select once on L6. No 900 or adaptive/grid expansion is allowed in V1.

Confirm the selected tau on the two frozen raw1000 halves separately and
pooled. Unknown official hull remains missing, never unstable.

### B2 — Freeze the stability target

For a matched candidate:

- delta Strict S.U.N. >=0 versus matched control;
- delta Meta S.U.N. >0 versus matched control;
- absolute requested1000 Strict S.U.N. >=10%;
- absolute requested1000 Meta S.U.N. >=50%;
- a headline replacement additionally may not underperform public Strict
  `105/1000`;
- body, Direct, novelty, uniqueness and stable-to-S.U.N. retention each no
  worse than -1 percentage point;
- both raw500 halves and both seeds reported; no selected-half headline.

### B3 — Noisy-state energy critic, only if B1/B2 are insufficient

Data:

- begin from the existing 1,752 Direct-valid CHGNet-labelled train-only
  structures, but rebuild labels if selected tau changes;
- retain exact-formula-disjoint train/validation split;
- use within-Plan continuous compatible-energy rank, not a hard stable token;
- unknown labels remain missing;
- novelty is evaluation-only.

Before training, require evaluator independence:

- held-out within-Plan Spearman lower confidence bound above zero;
- low/high energy AUC >0.60;
- agreement direction under MatterSim and official hull;
- no family/N-only shortcut.

The critic guides only the current legal discrete transition inside safe-axis.
It does not rerank completed samples. Use one frozen guidance coefficient and
report inference cost. Run two-seed L6 before any L7 confirmation.

### B4 — Relaxed-winner distillation

Run only if the critic is predictive but inference guidance is ineffective or
too expensive. Train CE on relaxed low-energy winners and stable MP-20 rows,
with frozen-reference KL. High-energy generated bodies never become CE
targets. Use one conservative update budget, no grid.

### B5 — RL last resort

RL requires a new explicit authorization after B0-B4. The historical TraceRL
is NO-GO. The minimum valid design must use a modern dLLM method such as
AGRPO/DiSPO, true online token-and-position behavior probabilities, legal
support renormalization, common refiner noise, same-formula group-relative
continuous rewards, reference KL/ESS/clip gates and independent MatterSim/
official evaluation.

RL is an engineering fallback, not automatically the second contribution.

## Ordered action items

- [x] Finalize job35518 and write the condition/schedule and raw/model494 reports.
- [x] Select full-axis or hard-axis strictly by the frozen two-seed gate: full-axis retained.
- [x] Decide whether the raw/model494 result triggers the fixed tau scan: triggered.
- [x] Freeze and execute CCFD Phase A0 on CPU with no GPU training.
- [x] Verify the current frozen tokenizer can host F1 without resizing or changing weights.
- [ ] Implement F1 online FSM on the current formula checkpoint without changing weights.
- [ ] Run F0/F1 two-seed requested1000 and evaluate correctness plus distribution drift.
- [ ] Run the fixed tau scan only if B1 triggers, then confirm one tau on raw1000.
- [ ] Audit critic evaluator independence only if the selected executor still misses stability targets.
- [ ] Train/evaluate one critic or one relaxed-winner candidate according to the sequential gates.
- [ ] Update BUILD_STATUS/PAPER_STORY, preserve public 105/488 until a complete replacement passes, test, commit and push.

## Validation

- Unit-test formula FSM reachability, exact N/Q conservation, mixed-valence,
  alloy/unary and dead-end accounting.
- Property-test 100% round-trip for every representable MP-20 formula.
- Verify requested denominators and no repair/replacement for every sampling arm.
- Recompute both legacy and independent composition correctness.
- Verify safe-axis schedule invariants and common-random-number ledgers.
- Report per-seed, per-half and pooled Strict/Meta stable, S.U.N., retention and
  energy quantiles.

## Terminal interpretation

- Track A pass + Track B pass: two clean contributions/effects in one system.
- Track A pass + Track B fail: composition contribution stands; public H1-A2
  remains the stability result.
- Track A fail + Track B pass: stability method/effect stands; no composition
  contribution claim.
- Both fail: retain H1-A2 public result and negative mechanism evidence; do not
  expand grids or relax gates.

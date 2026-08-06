# PlanGraph-DLM Experiment Charter V1

Status: `preregistered_design_no_execution_authorized`

Planning date: 2026-07-31

Method freeze deadline: 2026-08-31

Manuscript deadline: 2026-09-17

## 1. Research question

Can chemically structured planning and dependency-aligned discrete diffusion
raise crystal-generation validity and stable/unique/novel yield without
self-improvement, energy-label leakage, post-hoc repair, or changes to the
continuous diffusion refiner?

The central claim is not merely that an extra Planner helps. The claim is that
one explicit dependency representation can align:

- the Planner's output contract;
- the DLM's training corruption process; and
- the DLM's inference denoising order.

## 2. Hypotheses

### H-P: structured planning

A factorized `PlanGraph` will reduce parse, charge-neutrality, and
composition-rule failures relative to the frozen H1-A2 epoch-2 Planner while
preserving formula and plan diversity.

### H-D: dependency-aligned DLM

Mixed iid and planned corruption will reduce train/inference mismatch and
improve body completion and joint validity without degrading the fixed-panel
NLL by more than 1%.

### H-J: shared representation

Using the same PlanGraph for planning and denoising will outperform changing
only the Planner or only the DLM. The expected signature is a positive
interaction in the preregistered 2x2 comparison.

## 3. Scope and non-goals

In scope:

- a chemically factorized Planner output;
- planned DLM corruption and denoising;
- paired, all-attempt evaluation;
- benchmark-compatible accepted-N evaluation;
- explicit Planner-versus-DLM failure attribution.

Out of scope before the method freeze:

- self-training, online feedback, rejection-driven retraining, or iterative
  model self-improvement;
- S.U.N., CHGNet, MLIP, Materials Project, energy, or stability labels in
  training, checkpoint selection, repair, filtering, or reranking;
- modification of the continuous diffusion refiner;
- extra test-time search, retry, repair, filter, or rerank for primary
  all-attempt results;
- simultaneous architecture scaling.

The refiner may be studied only after the primary Planner/DLM method works and
only as a separately labeled post-freeze extension. It is not required for the
ICLR main claim.

## 4. Frozen and variable components

### Frozen throughout the primary study

- tokenizer and base model family;
- `dynamic_v1` crystal representation;
- train/validation split and fixed likelihood panel;
- H1-A2 epoch-2 baseline Planner;
- R5-C baseline body for the baseline arm;
- continuous diffusion refiner checkpoint and exact reverse-step settings;
- generation temperature and ordinal seed ledger;
- S.U.N. implementation and Materials Project reference snapshot;
- validity definitions and evaluation denominators.

### Allowed experimental variables

- Planner output schema: H1 plan versus PlanGraph;
- DLM corruption policy: iid versus planned mixture;
- DLM denoising group order derived from the registered dependency graph;
- planned-to-iid ratio, only in the registered ratio ablation.

### Development-split firewall

The local checkout currently contains an exact-length test mirror but not the
frozen train/validation JSONL. It may be used only to verify schema
convertibility, token coverage, and leakage invariants. Its chemical-bucket
distribution must not determine training mixture, thresholds, hyperparameters,
or checkpoint selection. Formal field coverage and all data-mixture decisions
must be recomputed on the frozen train split and registered before training.

## 5. PlanGraph contract

The Planner emits a schema-validated graph rather than free-form explanatory
text. The minimum fields are:

1. `composition`
   - reduced formula;
   - element counts;
   - candidate oxidation-state assignment;
   - net-charge check.
2. `symmetry`
   - crystal system or space-group constraint when supported;
   - symmetry confidence/status, never an energy proxy.
3. `lattice`
   - lattice family and bounded parameter relations.
4. `site_groups`
   - element identity and multiplicity;
   - dependency links to composition and symmetry;
   - deterministic generation order.
5. `constraints`
   - machine-checkable count, charge, and schema constraints;
   - no stability, formation-energy, or hull-distance field.

Every output must be reducible to the existing body prompt without injecting
ground-truth structure tokens. Unsupported fields cause a recorded Planner
failure; they are not repaired by a hidden second model.

For `plangraph_v1`, site groups are conservatively defined by element
multiplicity and the dynamic-v1 slot ledger. The current representation does
not carry reliable Wyckoff-equivalence labels, so v1 must not infer or import
them from ground-truth metadata. A symmetry-equivalent site grouping would be
a separately versioned future schema.

## 6. Planned corruption design

The existing answer layout is:

```text
[N] [a b c alpha beta gamma] ([element] [x] [y] [z]) * N
```

### D0: iid baseline

Use the current independent random-token masking process and current training
loss unchanged.

### D1: current-order alignment control

Use the current inference groups:

```text
N -> elements -> lattice -> x -> y -> z
```

For one planned-corruption example:

- sample one active group;
- keep prerequisite groups visible;
- stochastically mask tokens in the active group;
- fully mask later groups; and
- compute loss only on the active group.

D1 tests whether simple train/inference alignment helps, without claiming that
the order is chemically optimal.

### D2: PlanGraph dependency order

Use:

```text
N/composition -> symmetry/lattice -> site group 1 -> ... -> site group K
```

The site-group order is deterministic from the PlanGraph. In v1, the
composition group owns the atom-count and all element tokens, the
symmetry/lattice group owns the six lattice tokens, and each site group owns
the XYZ tokens of its registered slots. This makes composition locking
explicit and assigns every answer position to exactly one group. The same
visible-prerequisite, active-group, and masked-future rule used by D1 applies;
no ground-truth energy, stability, or hidden symmetry-equivalence property is
available.

### Mixture

The primary training mixture is:

```text
iid : planned = 2 : 1
```

The `1 : 2` mixture is a registered secondary ablation and is trained only if
the primary mixture passes the likelihood and numerical-stability preflight.
The iid and planned examples come from the same frozen supervised corpus; no
generated sample is fed back into training.

### Schedule choice

D2 is the proposed method. D1 is a mechanistic control. If D2 cannot be
implemented without changing the representation or leaking unavailable
information by 2026-08-10, D1 becomes the bounded DLM method for this
submission and the broader PlanGraph schedule is deferred.

## 7. Registered arms

### Planner-only screen: 512 attempts per arm

| Arm | Planner | Purpose |
|---|---|---|
| P0 | frozen H1-A2 epoch 2 | baseline |
| PG | PlanGraph Planner | structured-planning test |
| PG-shuffle | PG fields with dependency links shuffled | mechanism control |

All arms use the same ordinal seeds and the first 512 attempts. There is no
sample-id selection, retry, replacement, repair, filter, or rerank.

### DLM pre-screen

| Arm | Planner input | DLM corruption | Purpose |
|---|---|---|---|
| D0 | frozen ground-truth plan panel | iid | baseline |
| D1 | same panel | iid:current-order = 2:1 | alignment control |
| D2 | same panel | iid:PlanGraph = 2:1 | proposed corruption |
| D2-ratio | same panel | iid:PlanGraph = 1:2 | ratio ablation |
| D2-shuffle | shuffled dependency order | iid:planned = 2:1 | mechanism control |

D2-ratio runs only after D2 passes preflight. D2-shuffle may use a shorter
budget because its role is falsification, not model selection.

### Primary 2x2 study

Let `D*` be D2, or D1 only under the preregistered feasibility fallback.

| Arm | Planner | DLM | Refiner |
|---|---|---|---|
| M0 | H1 P0 | D0 iid baseline | frozen |
| M1 | PlanGraph | D0 iid baseline | frozen |
| M2 | H1 P0 | D* planned mixture | frozen |
| M3 | PlanGraph | D* planned mixture | frozen |

M1 estimates the Planner contribution, M2 estimates the DLM contribution, and
M3 tests their joint effect. M0 is always evaluated from the frozen identity;
it is never silently replaced by a newly trained approximation.

## 8. Evaluation protocol

### Protocol A: raw all-attempt yield

- use the first registered `N` ordinal seeds;
- every attempt remains in the denominator;
- no retry, replacement, repair, filtering, or reranking;
- report parser failure, Planner chemistry failure, body/CIF failure,
  composition mismatch, refiner failure, and evaluator failure separately.

This is the primary systems-yield protocol.

### Protocol B: benchmark-compatible accepted-N

- reject only the same parser/gross-invalid pre-generation cases excluded by
  the reference benchmark;
- sample sequential registered seeds until `N` accepted generations;
- never replace a structure for being invalid, unstable, non-unique, or
  non-novel after acceptance;
- cap total attempts at `4N`; failure to reach `N` is itself reported;
- report the raw number of attempts required and all rejection reasons.

This protocol supports comparison with prior CrysLLMGen-style results. It must
be shown next to Protocol A, never instead of it.

### S.U.N. definitions

For both protocols:

```text
strict S.U.N. = stable AND unique AND novel / registered denominator
meta S.U.N.   = metastable AND unique AND novel / registered denominator
```

The Materials Project API/cache version and energy thresholds are frozen
before evaluation. API credentials are runtime secrets and never enter source,
logs, manifests, or reports.

## 9. Metrics

### Planner

- schema parse rate;
- plan completion rate;
- composition validity;
- charge-neutrality and Pauling-rule failure counts;
- unique-formula rate;
- element-count, crystal-system, and formula-distribution drift.

### DLM/generation

- fixed-panel NLL relative to initialization;
- finite-loss and finite-gradient checks;
- body eligibility and body completion;
- CIF parse and structure validity;
- composition agreement between plan and generated structure;
- joint validity and generation completion;
- unique and novel rates.

### Final scientific outcomes

- strict and metastable S.U.N.;
- paired differences M1-M0, M2-M0, and M3-M0;
- factorial interaction `(M3-M2) - (M1-M0)`;
- paired bootstrap confidence intervals;
- McNemar tests for binary paired outcomes;
- across-seed mean, median, range, and per-seed values.

No conclusion is based on a point estimate alone.

## 10. Decision gates

### G0: implementation integrity

Required before training:

- exact deterministic dependency groups for a fixed seed;
- every answer token assigned to exactly one group;
- prerequisites are never masked in a planned example;
- future groups are fully masked;
- loss is restricted to the active group;
- iid behavior is bitwise unchanged when planned mixing is disabled;
- unit tests prove that S.U.N., CHGNet, MP, `e_above_hull`,
  `formation_energy`, and other energy fields never enter prompts, losses,
  sampling, or checkpoint selection;
- all H1 SHA checks still pass.

Failure: stop and fix locally; do not submit training.

### G1: Planner-only 512

PG continues only if:

- parse rate is at least 99%;
- plan completion is at least 98%;
- composition validity is at least 95%;
- unique-formula rate is at least 95% of P0;
- no major formula or crystal-system distribution drift is unexplained; and
- PG beats PG-shuffle on composition validity.

Failure: keep H1 Planner, diagnose the schema/data target, and do not compensate
with filtering.

### G2: DLM likelihood preflight

D1/D2 continues only if:

- no OOM, NaN, infinite loss, broken gradient, or schedule violation occurs;
- fixed-panel NLL is no worse than 1% relative to the frozen initialization;
- the direct paired planned-margin diagnostic is positive; and
- validation is evaluated every 50 updates, with at most 400 screening
  updates before checkpoint selection.

Checkpoint selection uses only supervised likelihood and registered direct
validity diagnostics, never S.U.N. or energy.

### G3: paired-256 screen

All four M arms use the same 256 ordinal seeds. M3 advances if:

- generation completion is at least 97%;
- no arm loses more than 2 percentage points of structure validity versus M0;
- M3 improves composition validity by at least 5 points versus M0;
- M3 improves strict S.U.N. by at least 1 point or crosses 10%;
- M3 improves metastable S.U.N. by at least 3 points or crosses 50%;
- M3 beats at least one of M1/M2 and the interaction estimate is not strongly
  negative; and
- the likelihood gate still holds.

This is a screening gate, not a publication claim. Confidence intervals and
all failures remain visible.

### G4: three-seed confirmation

Run 3 seeds x 1,000 attempts for M0 and the best preregistered candidate. The
candidate is considered stable enough for the main paper only if:

- pooled raw all-attempt strict S.U.N. exceeds 10%;
- pooled raw all-attempt metastable S.U.N. exceeds 50%;
- at least two of three seeds individually cross both 10% and 50%;
- no seed falls below 9% strict or 48% metastable;
- the paired 95% bootstrap confidence interval for both S.U.N. deltas excludes
  a negative effect;
- composition, structure, joint validity, diversity, and NLL gates hold; and
- Protocol B reaches its accepted-N target without exceeding the attempt cap.

If G4 fails by 2026-08-31, activate the H1 fallback rather than adding a new
major component.

### G5: final-scale evidence

Only after G4:

- run accepted-N = 10,000 for M0 and the candidate if resources permit;
- otherwise freeze the three-seed confirmation as the final quantitative
  result and state the scale limitation;
- no method or metric changes after seeing final-scale outcomes.

## 11. Failure attribution

Every all-attempt record receives exactly one earliest failure label:

1. `planner_parse`;
2. `planner_schema`;
3. `planner_chemistry`;
4. `body_ineligible`;
5. `dlm_decode_or_cif`;
6. `composition_mismatch`;
7. `refiner`;
8. `evaluation`;
9. `completed`.

Planner-caused and DLM-caused composition failures are reported separately.
The first valid failure label is immutable; later diagnostics may be attached
but may not reassign the denominator.

## 12. Timeline

| Dates | Deliverable | Hard decision |
|---|---|---|
| Jul 31-Aug 2 | freeze charter, H1 fallback, PlanGraph schema, metric code | no training without G0 plan |
| Aug 3-Aug 10 | PlanGraph targets, Planner implementation, plan-only 512 | PG passes G1 or H1 Planner remains |
| Aug 11-Aug 21 | D1/D2 corruption, tests, <=400-update screens | choose D* under G2 |
| Aug 22-Aug 28 | primary 2x2 paired-256 | M3 passes G3 or stop |
| Aug 29-Aug 31 | three-seed confirmation launch/result and method freeze | G4 path or H1 fallback |
| Sep 1-Sep 5 | final experiments and first complete manuscript | complete draft by Sep 5 |
| Sep 6-Sep 12 | analysis, ablations, figures, internal review | no new major method |
| Sep 13-Sep 14 | submission-ready revision | freeze claims and tables |
| Sep 15-Sep 16 | upload dry run and reproducibility check | final package |
| Sep 17 | buffer and submission | no experiment dependence |

## 13. ICLR narrative if successful

The paper should emphasize three linked contributions:

1. a chemically structured planning interface that exposes and reduces
   upstream failures;
2. a dependency-aligned discrete diffusion objective that closes the
   train/inference mismatch; and
3. a transparent dual evaluation protocol that separates benchmark
   comparability from raw end-to-end yield.

The refiner is deliberately frozen. This is a strength of the causal story:
improvements can be attributed to planning and DLM rather than to an
uncontrolled stack of changes.

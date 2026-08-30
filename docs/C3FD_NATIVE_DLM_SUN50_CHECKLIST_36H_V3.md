# C3FD-native DLM stability checklist v3

Date: 2026-08-30  
Deadline: 2026-08-31 23:30 Asia/Shanghai  
Status: active execution contract

## Outcome target

The final prospective, fixed-denominator experiment targets:

- **Strict S.U.N. >= 10.0%**;
- **Meta S.U.N. >= 50.0%**.

For a requested denominator of 256, these correspond to at least 26 Strict
S.U.N. and 128 Meta S.U.N. per 256-attempt rate before stream averaging. For
each independently trained policy seed, the primary rate is the mean of its two
predeclared process streams on the same 256 compositions. Both policy seeds,
all four streams, and composition-cluster confidence intervals are disclosed.
The targets are objectives, not result-deletion rules: missing either target is
reported as a negative or partial result and does not authorize seed,
checkpoint, temperature, tau, cohort, or denominator selection.

Historical references are reported beside, but never substituted for, the
prospective result:

- H1-A2 historical compatibility: `9.40/47.40%` Strict/Meta;
- corrected H1-A2 exact: `8.58/46.08%`;
- corrected H1-A2 continuous: `7.63/45.47%`;
- R03 process high points are mechanism references, not independent Planner or
  training-seed replications.

## Paper-facing system

The active story returns to one native modular pipeline:

1. **C3FD semantic Planner** proposes a certificate-carrying, exactly valid
   composition and one coarse structural Plan.
2. **Planner-native masked DLM** consumes that same Plan schema and generates
   every lattice and atomic coordinate token under exact `7+4N` cardinality.
3. **Continuous diffusion refinement** performs geometry projection with
   model494 tau800 while preserving composition.

The old H1-A2 rich prompt is now only a compatibility diagnostic. The paper
method does not continue either the old rich or minimal adapter. Its
`prototype_key`, `oxidation_candidates=unknown`,
and legacy JSON semantics are not the production interface.

## `C3FD_NATIVE_PLAN_V2`

The production prompt is a compact, typed serialization of fields already
emitted or certified by the current Planner.

Portable hard conditions:

- `N`;
- ordered `elements` and exact `counts`;
- `anion_framework` / broad family, deterministically derived from composition.

Soft structural hints:

- `lattice_system`;
- `spacegroup_bucket`;
- `volume_per_atom_bin`.

The DLM output remains the unchanged dynamic body:

```text
N, a, b, c, alpha, beta, gamma,
element_1 ... element_N,
x_1 ... x_N,
y_1 ... y_N,
z_1 ... z_N
```

Planner certificates remain internal to C3FD and are not serialized into the
DLM prompt. There is no valence, charge-bucket, prototype, or dataset-specific
certificate field in the public interface. Hard composition/cardinality fields
are always visible and exact. Soft fields
are hints, never hard geometry constraints. Training applies a frozen,
train-validation-calibrated field dropout/corruption policy so 60--70%-accurate
Planner fields cannot force the DLM into an incorrect basin. No new full-vocab
resize, prototype lookup, target Wyckoff positions, oracle space group, or
direct Planner CIF is introduced.

## Data contract

### A. Planner-interface adaptation SFT

Use all rows in the original MP20 standard train (`27,136`) and validation
(`9,047`) splits. MP20 crystal structures themselves are the SFT supervision;
no row is filtered by a Planner certificate. MP20 is split by material/structure
entry rather than by chemsys and has `3,469` overlapping chemsys. This overlap
predates native rendering and is disclosed rather than treated as an error.
Consequently this run calls the split **MP20 standard
validation**, never chemsys-held-out validation. Each MP20 crystal contributes
exactly one ground-truth teacher rich JSON prompt and one crystal-body target.
C3FD-predicted Plans are inference inputs only and are never mixed into SFT.
Energy, hull, prospective outcomes, and current development-canary outcomes are
excluded from this SFT dataset. Legacy source
rows use immutable file ordinal for alignment; composition, N, and answer
alignment against semantic and dual-Planner rows must remain exact.

### B. Stability alignment data

The existing 3,614 historical candidates are **development-only evidence**.
They combine retired L6/L7/D3PO sources, contain upstream raw-validity
selection, and cannot support the main paper training claim or a new
confirmatory test.

If Planner-native SFT restores execution but does not reach the stability
target, create a new on-policy, train-only pool from the frozen SFT checkpoint:

- freeze a set of MP20-train compositions and a fixed `K` before sampling;
- exactly `K` trajectories per composition with common seed ledgers;
- preserve every raw-invalid and failed attempt;
- evaluate raw Direct, raw CHGNet when known, and model494-refined CHGNet;
- compare energies only within one composition;
- keep group total weight one;
- raw-invalid is lexicographically worse than every raw-valid candidate;
- the denoising anchor is `best_valid_candidate_index`, never simply the
  lowest post-refiner energy;
- no candidate reranking or best-of-N is used at final inference.

This is described, if successful, as on-policy same-composition energy
alignment of a Planner-conditioned DLM. It is promoted to a contribution only
after prospective two-training-seed replication.

## Training contract

### Stage 1: fresh Planner-native DLM training

- initialize a new LoRA adapter from the shared pretrained LLaDA-8B backbone;
  do not load the old rich or minimal adapter into the paper method;
- keep the recovered H1-A2 LoRA structure: `r=8`, alpha `32`, dropout `0.05`,
  target modules `q/k/v/ff/up`;
- two independent DLM training seeds;
- each of the 27,136 MP20 teacher-rich rows contributes once per source epoch;
- epoch1: 1,696 optimizer updates, effective batch 16, LR `5e-5`, cosine,
  warmup 100, minimum LR ratio `0.2`;
- epoch2: another 1,696 updates, LR `1e-5`, cosine, warmup 100, minimum LR
  ratio `0.1`;
- total: 3,392 optimizer updates; only the epoch2/step3392 adapter is eligible
  for prospective evaluation. Epoch1/step1696 is monitoring-only;
- SFT is ordinary teacher-rich masked-body CE, matching the original H1-A2
  training story; Planner prediction error is evaluated at inference time;
- no early stopping, epoch/checkpoint selection, or seed selection.

### Stage 2: optional stability alignment

Run only from the frozen Stage-1 checkpoint and fresh pool:

```text
L = L_native_body_CE
  + lambda_rank * L_same_composition_continuous_rank
  + lambda_raw * L_raw_validity_gated_rank
  + lambda_ref * L_quadratic_reference_bound
  + lambda_anchor * L_best_valid_anchor
```

Constants are calibrated once from train-only gradient norms and frozen; there
is no test-driven grid. If no valid candidate exists in a group, the group is
handled by one predeclared fail-closed rule. The current energy-only wrapper is
forbidden because it can reward a raw-invalid structure whose refiner outcome
has low energy.

## Prospective evaluation

- freeze one new outcome-blind C3FD Planner-seed ledger of 256 compositions
  before policy outcomes;
- verify exact-composition disjointness from MP20 train, D3PO main/sealed,
  L6/L7, rich seed19 development, and old H1 cohorts;
- BASE and two policy seeds, each with streams17/18;
- one Plan and one trajectory per attempt;
- temperature `0.7`, exact-axis schedule, model494 tau800;
- one six-cell generation, one raw-first/refined 12-cell evaluation, and one
  fresh official MP query;
- the already supplied MP credential is injected only into that single query's
  temporary non-ambient process environment, unset immediately after launch,
  and never written to repository files, automation prompts, commands, logs,
  hashes, or manifests; it is not requested again during this task;
- requested denominator 256, with missing attempts retained by sample index;
- unknown official hull remains missing, never relabeled unstable;
- report raw/refined CHGNet, official `e_hull` ECDF/quantiles, Direct/N/U/NU,
  Strict/Meta stable and S.U.N., seen/unseen chemsys, McNemar, and paired
  composition-cluster bootstrap intervals.

## Decision and fallback tree

1. **Faithful H0/R0S diagnosis** explains the historical regression and runtime
   compatibility only; it cannot change the fresh-LLaDA initialization of the
   paper method and is not a final claim.
2. **Planner-native SFT reaches 10/50:** freeze it as the primary method and do
   not open stability-alignment training merely to increase the headline.
3. **SFT restores H1-A2-level execution but misses 10/50:** run exactly one
   fresh on-policy same-composition alignment contract.
4. **SFT loses raw execution:** diagnose native serialization/soft-field
   brittleness on train/validation; do not hide it with model494 or tune the
   prospective cohort.
5. **Alignment improves only post-refiner:** classify it as refiner-mediated;
   do not claim the raw DLM learned stability.
6. **Two training seeds disagree:** report unstable/negative; no seed choice.

Pure AR, external oracle rich Plans, direct Planner CIF, intent heads, target
SG/Wyckoff/prototype, composition tilting, survivor filters, reranking,
replacement, best-of-N, tau/temperature/checkpoint sweeps, RL/GRPO/SMC, and
test-outcome training are out of scope.

## Resource schedule

| Phase | Maximum resource | Expected wall time | Notes |
|---|---:|---:|---|
| faithful H0/R0S offline completion | 4 A800, 32 CPU | remaining 0.5--1.5 h | job38603 already running; development only |
| native Plan/data build and archive | 0 GPU, 16 CPU | 0.25--0.75 h | MP20-standard split; immutable hashes |
| fresh trainer/wrapper implementation | 0 GPU, <=32 CPU | 1.5--3 h | teacher-rich only, two-stage LR |
| two-seed fresh native SFT | 4 A800, 32 CPU | 1.5--2.5 h | two matched 2-A800 jobs, step3392 only |
| train/standard-val raw-first canary | <=6 A800, 48 CPU | 1.5--3 h | diagnostic only; no seed choice |
| fresh on-policy pool plus alignment, if used | <=6 A800, 48 CPU | additional 5--8 h | fixed K and one frozen setting |
| prospective generation | 6 A800, 48 CPU | 1--1.5 h | six cells once |
| raw/refined evaluation | 6 A800, 48 CPU | 2--3 h | 12 cells once |
| official query and scientific finalizer | 0 GPU, <=8 CPU | 0.5--2 h | one immutable union/query |
| final report/story freeze | 0 GPU, <=16 CPU | 1--2 h | RQs, claims, archives, paper story |

From 2026-08-31 00:30 Asia/Shanghai, the no-alignment critical path is expected
to finish all S.U.N. results and the paper storyline around `14:00--18:00`. If
the single permitted fresh alignment route is scientifically needed, the
expected completion moves to `20:00--23:00`, still before the 23:30 deadline but
with little engineering slack. These are planning ranges, not result gates.

CPU and MP query latency do not consume the GPU ceiling. At most two jobs may
be active or pending, and aggregate GPU allocation may not exceed six A800s.

## Execution checklist

- [x] Historical H1-A2/R03/R5C/minimal/SGTC/D3PO evidence audit complete.
- [x] Rich prompt null-schema bug reproduced and regression-tested.
- [x] Immutable faithful H0/R0S v2 cohort frozen with accounting metadata.
- [x] Faithful H0/R0S generation terminal.
- [ ] Faithful raw/refined offline diagnosis job38603 completed; finalizer and
  archive remain. It cannot alter fresh paper-method initialization.
- [x] `C3FD_NATIVE_PLAN_V2` rich-JSON serializer/round-trip/type tests complete;
  Planner certificates are not part of the DLM interface.
- [ ] Full MP20 standard train/validation teacher-rich-only V2 data frozen.
  Jobs38681/38684 are valid but superseded multi-view development artifacts and
  are not trainable.
- [x] Dual-C3FD predicted soft-field coverage reported: train `27,136`,
  validation `9,047`, both checkpoints and all three fields at 100% coverage.
- [ ] Two-seed fresh Planner-native SFT terminal with only step3392 adapters;
  epoch1/step1696 diagnostics disclosed but never selected.
- [ ] Train/validation raw execution and stability canary terminal.
- [ ] Decision to stop at SFT or open one fresh on-policy pool frozen.
- [ ] If opened, fresh pool and safety-aware K-way trainer terminal.
- [ ] New prospective C3FD ledger frozen before policy outcomes.
- [ ] Prospective six-cell generation terminal.
- [ ] Prospective raw/refined 12-cell evaluation terminal.
- [ ] One fresh official MP query and finalizer terminal.
- [ ] `SUN >= 10%` and `Meta S.U.N. >= 50%` evaluated without denominator or
  seed selection.
- [ ] Positive and negative archives, resource accounting, BUILD_STATUS,
  PAPER_STORY, tests, commit, and push complete.
- [ ] Final RQs and 2--3 contributions classified as SUPPORTED, CANDIDATE, or
  UNSUPPORTED with forbidden claims listed.

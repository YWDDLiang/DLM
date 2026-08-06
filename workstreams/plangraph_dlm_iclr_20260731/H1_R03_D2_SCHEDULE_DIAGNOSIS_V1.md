# H1-A2 R03 D2 Schedule Diagnosis V1

Date: 2026-08-02
Disposition: read-only diagnosis; no rerun, repair, checkpoint change, or
downstream evaluation was performed.

## Executive conclusion

R03 failed because the D2 decoding schedule broke a safety invariant that the
frozen H1 decoder implicitly relies on:

> every active site's X and Y coordinates must be known before any Z
> coordinate is committed.

The H1 D1 schedule enforces that invariant globally (`all X -> all Y -> all
Z`). D2 instead places all XYZ positions for one element-multiplicity group
into a common confidence-ranked group. Inside that group, X, Y, and Z may be
committed in any order. The frozen duplicate-coordinate mask only constrains Z
logits, and it can do so only when that site's X and Y are already known. Once
Z is committed early, later X or Y decisions can complete an exact duplicate,
but neither X nor Y is protected by the mask.

This is principally a schedule/constraint interface incompatibility. It is
not evidence that the P0 Planner, B0 checkpoint, tokenizer, input Plan, seed
ledger, or GPU execution degraded. The current D2 treatment is scientifically
rejected. H1 D1 remains the baseline.

The actual per-token reveal trace was not recorded, so the diagnosis cannot
name the exact step at which each Z token was committed. Nevertheless, the
paired outcomes, Z-specific collapse, collision topology, and deterministic
code path all point to the same mechanism.

## 1. Isolation quality

R03 is a clean schedule-only comparison:

- Planner: frozen H1-A2 P0 in both arms.
- Body model: the same loaded frozen B0/R5-C adapter in both arms.
- Inputs: identical Plans and compiled prompts; paired input mismatch count 0.
- Sampling: identical ordinal body-noise seeds.
- Execution: identical shared batch partition in one Slurm job.
- Treatment: only D1 versus D2 generation-position groups.
- Denominator: 32 raw attempts per arm.
- No retry, replacement, repair, filtering, or reranking.
- No refinement, direct metric, S.U.N., training, or downstream action.
- Job 29669 completed `0:0`; no system-error signature was found.

Therefore the 17 paired completion losses are attributable to the schedule
treatment, not to an uncontrolled model or data change.

## 2. Paired result

| Outcome | Count |
|---|---:|
| D1 succeeded, D2 succeeded | 14 |
| D1 succeeded, D2 failed | 17 |
| D1 failed, D2 failed | 1 |
| D1 failed, D2 succeeded | 0 |

- D1: `31/32 = 96.875%`.
- D2: `14/32 = 43.750%`.
- D2 failure rate: `18/32 = 56.250%`; Wilson 95% interval
  `[39.33%, 71.83%]`.
- All 18 D2 failures are
  `body:DuplicateCoordinateError`.
- Exact paired McNemar p-value: `1.52587890625e-05`.
- All 18 failed texts contained the full planned atom count and parsed into
  complete token-level bodies before graph construction.
- D1 contained zero exact coordinate-collision pairs over all 32 outputs.
- D2 contained 93 exact coordinate-collision pairs.

The failure occurs at `body_graph`, after body generation and Plan matching.
It is not a Planner parse/composition failure.

## 3. Collision topology

The 93 D2 duplicate pairs are not confined to one bad site-group definition:

| Topology | Duplicate pairs | Failed samples containing topology |
|---|---:|---:|
| Within the same element/PlanGraph group | 51 | 11/18 |
| Across different element/PlanGraph groups | 42 | 13/18 |

Some samples contain both topologies. Cross-group collisions in 13/18 failed
samples are particularly important: they show that the problem is not merely
"the model makes equivalent sites identical." Global collision protection is
being bypassed while groups are decoded sequentially.

Examples:

- ordinal 0, `Se8Ta4`: two cross-group Se/Ta collisions;
- ordinal 16, counts `[2,14,2]`: 23 duplicate pairs, including within-F and
  cross Li/F/Cu collisions;
- ordinal 21, counts `[2,1,2]`: a three-species collision at `(0,0,0)` even
  though the largest D2 group has only six coordinate tokens;
- ordinal 30, counts `[2,2]`: two cross-group P/V collisions with a largest
  D2 group of only six tokens.

Thus large groups can aggravate the failure, but are not required for it.

## 4. Axis-specific output signature

Mean per-sample fraction of unique coordinate values:

| Axis | D1 control | D2 candidate | Difference |
|---|---:|---:|---:|
| X | 0.27490 | 0.27414 | -0.00075 |
| Y | 0.29511 | 0.29268 | -0.00243 |
| Z | 0.65496 | 0.51105 | **-0.14391** |

Mean repeated-XY pair counts are similar: D1 `25.94`, D2 `27.16`. X and Y
support therefore did not generally collapse. Z support did. This is exactly
the axis on which the current uniqueness mask operates, and exactly the axis
whose safety depends on X/Y already being visible.

## 5. Deterministic code path

### D1 preserves the mask precondition

`current_order_groups` creates:

1. atom count;
2. elements;
3. lattice;
4. all X positions;
5. all Y positions;
6. all Z positions.

Count and elements are prefilled in R03. By the time any Z is selected, every
active X/Y pair is known. For a masked Z, the duplicate mask can enumerate all
already committed Z values sharing that X/Y and ban them.

### D2 removes the mask precondition

`plangraph_dependency_groups` creates one coordinate group per
element-multiplicity site group and puts every slot's X, Y, and Z into that
same group. The PlanGraph groups themselves are constructed only from
element identity and multiplicity; they are not inferred Wyckoff or geometric
equivalence classes.

The paired generator:

1. evaluates all still-masked positions in the active group;
2. ranks them by model confidence;
3. commits the top position;
4. repeats until the group is exhausted.

For the homogeneous R03 batches, group steps equal the number of masked
positions, so this is one committed token per row per step. There is no axis
ordering constraint. A Z token can therefore be selected before its own X or
Y.

### The duplicate mask cannot recover

The mask:

- reads X and Y for each slot;
- skips the slot if either is still masked/unknown;
- builds previously seen Z bins by X/Y;
- masks only the target Z logits.

It never masks X or Y logits. If Z is committed first, a later X or Y token
can complete a duplicate coordinate without encountering the guard. Final
graph construction then rejects the duplicate.

The final graph validator correctly rejects any exact/PBC-equivalent
fractional coordinate, including cross-species collisions. The validator is
exposing the schedule error; it is not causing the scientific degradation.

Relevant frozen sources:

- `execution/h1_body_schedule32_v1/runtime/crystal_dlm/planned_corruption.py`
  lines 247-302;
- `execution/h1_body_schedule32_v1/paired_llada.py` lines 164-222;
- `execution/h1_body_schedule32_v1/runtime/crystal_dlm/llada_generation.py`
  lines 253-341;
- `execution/h1_body_schedule32_v1/runtime/crystal_dlm/plangraph_v1.py`
  lines 399-443;
- `execution/h1_body_schedule32_v1/runtime/scripts/sample_llada_dynamic_crystals.py`
  lines 396-404.

## 6. Group width is secondary, not the root cause

- Largest D2 coordinate-group width at least 18:
  `10/15 = 66.7%` failed.
- Width below 18: `8/17 = 47.1%` failed.
- Mean maximum width: failed `19.33`, succeeded `17.36`.

This is a modest risk increase, not a monotone relationship:

- width 30 failed `4/4`;
- width 36 passed `4/4`;
- width 6 still failed `3/8`.

The hard invariant violation explains failures at every size. Wider mixed-axis
groups merely provide more opportunities for an unsafe reveal order.

## 7. Secondary contributors

### Train/inference schedule mismatch

B0 was not trained to use D2's element-group mixed-axis reveal order. D2
changes the conditional contexts seen during denoising. The frozen B0 model
therefore assigns confidence under a context ordering it was not optimized
for. This likely contributes to early, low-diversity Z commitments.

Training alone is not an acceptable fix for the hard guard violation. A model
must not be expected to learn an implicit reveal order merely to satisfy a
geometric uniqueness invariant.

### PlanGraph semantics are currently coarse

The current "site groups" are element-multiplicity blocks derived from the
Plan's composition. No generated geometry or true Wyckoff relation is
available at this point. They should be described as
composition-conditioned coordinate blocks, not crystallographic
equivalence groups. This coarseness can reduce the value of the dependency
schedule, but it does not by itself explain the cross-group failures.

## 8. Rejected alternative explanations

- **Planner degradation:** impossible in this comparison; both arms use the
  same frozen P0 Plans, and body Plan matching passed.
- **B0 checkpoint or tokenizer mismatch:** both arms use one loaded model
  instance and the same frozen tokenizer.
- **Seed or batch mismatch:** paired seeds and shared batch partition are
  exact.
- **Parser/length failure:** every D2 failure has a complete N-matching body.
- **GPU/system failure:** the job completed normally and no CUDA/OOM/NCCL/NaN
  signature exists.
- **Only large compositions fail:** false; failures occur at N=4 and D2 group
  width 6.
- **Final validator is too strict:** false for this protocol; exact duplicate
  fractional coordinates are invalid inputs to the downstream graph/refiner,
  and D1 avoids them under the same validator.

## 9. Safest next single-variable candidate

Do not repair or relabel D2. Register a new schedule, tentatively
`D2-safe-axis`, with this order:

1. frozen composition prefill;
2. lattice;
3. X coordinates split by PlanGraph element group;
4. Y coordinates split by PlanGraph element group;
5. Z coordinates split by PlanGraph element group.

This keeps the only useful PlanGraph intervention—group-aware ordering within
each axis—while restoring the H1 invariant that all X/Y values exist before
any Z.

Before one GPU attempt, require CPU/unit gates:

- every active X and Y position precedes every active Z position;
- every scheduled Z step has a known X/Y pair;
- synthetic same-group and cross-group collision fixtures are blocked;
- reveal-step logging records `z_before_xy_count=0`;
- the frozen D1 schedule remains byte-identical.

Then run exactly one new paired-32 schedule-only screen:

- `P0+B0+D1` versus `P0+B0+D2-safe-axis`;
- same 32 ordinals, prompts, batch partition, and stateless noise;
- no training, refinement, direct metric, or S.U.N.;
- gate: at least `31/32` completion, zero excess duplicate-coordinate
  failures, and no new failure class.

Do not change the duplicate mask and schedule in the same experiment. Do not
train a planned-corruption checkpoint until the safe schedule itself passes
on B0. If the safe schedule passes, planned training can become a later,
separate H1-based factor using the already frozen one-epoch/LR `5e-5`
contract.

## 10. Evidence

- Run:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_schedule32_v1`
- Job: 29669, `COMPLETED 0:0`.
- Terminal SHA-256:
  `898191bfe23b66ecf811eb8b223d1a7356181b273a8468c2439a926d213b09e3`.
- Generation report SHA-256:
  `fdc9aa4939274a2a44173358f7735f53a7bd7648a161c31ec979bfbc50991835`.
- Control attempts SHA-256:
  `146a1eaac959c176ff740bad1daba8ad5ae791f306f50fe565a75756ff02fe0f`.
- Candidate attempts SHA-256:
  `61de580e8aebc1c841e6da9c340a74fd7db41050f59e2e54baeff2a0966cc503`.
- Read-only diagnostic script:
  `analysis/r03_schedule_diagnosis.py`.
- Diagnostic script SHA-256:
  `085c0fb4e52f2d1fddfda3ab2f7e502b53b4b57fe5a76fbb3ac2114c4eb6918a`.

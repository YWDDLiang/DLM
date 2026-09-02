# Rollout-Matched DLM 24-hour checklist V1

Deadline: 2026-09-03 23:30 Asia/Shanghai.

## Current state (2026-09-02)

- Corrected immutable cohort:
  `rollout_matched_pilot_128x128_v2_20260902`.
- It contains 128 rollout-train and 128 untouched holdout rows, with zero exact
  composition overlap and deployment-mode `C3FD-predicted-native-V2` prompts.
- The earlier V1 cohort and capture job 39253 used teacher prompts. Job 39253
  was cancelled before training; both are retained only as engineering traces.
- Corrected exact-axis capture job 39259 completed in 13:43 on one A800 and
  produced 512 transitions (384 train / 128 validation). No pilot training was
  submitted.
- The geometry execution interface is specified in
  `TOKEN_NATIVE_PBC_GEOMETRY_EXECUTOR_V1.md`; implementation is deferred until
  the canonical-site DLM establishes whether an explicit stability score is
  still needed.
- A full-data audit found that only 14.995%/14.325% of train/validation teacher
  bodies use the same element-slot order as inference hard prefill. The primary
  next route is therefore canonical MP20 site ordering, not another CE loss.
- Canonical data V2 is terminal at `27136/9047` rows with zero drops and zero
  prompt changes; token audit has zero truncation and zero prompt/answer boundary
  mismatches. Job 39280 failed before step0 because it referenced the stale
  trainer root; parameter-identical recovery job 39282 completed successfully
  in `01:18:39` on two A800.
  Its run config pins fresh initialization, canonical V2 data, seed82017 and
  3392 updates; `fresh_lora=true`, `exact_zero_delta=true` and
  `lora_B_max_abs=0.0` have passed before the first optimizer update. Formal
  training remained finite through the fixed endpoint. Validation loss was
  `2.226879` at monitoring-only step1696 and `2.383175` at eligible step3392;
  the latter is retained without checkpoint selection.

## Objective

Improve raw periodic geometry and stability after C3FD has already solved
composition validity. Planner, composition distribution, dynamic `7+4N`,
exact-axis inference, temperature and model494 tau800 remain frozen.

The pilot tests whether training on the DLM's own committed exact-axis errors
improves continuation geometry. It is not another synthetic planned-corruption
run: historical B3 improved synthetic NLL but worsened actual-rollout NLL and
Strict S.U.N.

Supervision is MP20-only. A real rollout supplies the erroneous input state,
while the paired original MP20 body is the sole teacher; generated, refined,
CHGNet-selected or hull-selected structures never become labels.

## Frozen prior evidence

- G0: every parsed structural failure was a PBC collision; token round-trip
  caused zero validity flips.
- G2: small real geometry benefit, but raw stability remains weak.
- B3 synthetic safe-axis corruption: terminal negative; do not repeat.
- BTRD and synthetic Force residual: terminal negative; do not extend.
- Force microstudent holdout: 86/128 texts changed, but Direct flips were
  `6 invalid→valid` and `6 valid→invalid`, for net zero.
- Joint/mixed/atom-major inference schedules, extra ordinary CE, SGTC,
  post-refiner-only D3PO, remasking and tau sweeps remain closed.

## P0 — outcome-blind Plan freeze

- [x] Select 256 unique MP20-train rows with deterministic N/arity stratification.
- [x] Split by source row into 128 rollout-train and 128 untouched holdout Plans.
- [x] Preserve deployment C3FD-predicted V2 prompt, exact `N`, element counts,
  target dynamic body,
  source-row provenance and fixed sample order.
- [x] Read no CHGNet, hull, Direct, prospective or historical result during selection.

## P1 — real exact-axis rollout capture

- [x] Freeze promoted G2-A, sampling seed 95117 and capture contract.
- [x] Keep exact-axis order: lattice, all X, all Y, all Z.
- [x] Capture one state inside each of the four active groups per train Plan.
- [x] Preserve every previously committed model token, including errors.
- [x] Pair every generated input state only with its original MP20 target body.
- [x] Fill masked suffix positions only for storage; record the original model
  mask and the active-group supervision mask separately.
- [x] Job 39259 produced exactly `128 × 4 = 512` transition rows; no retries or
  replacement.

## P1.5 — teacher-continuation realizability audit

- [x] Treat each captured `source_answer` as an oracle continuation: preserve
  every committed rollout token and fill only still-masked positions with the
  paired original MP20 body.
- [x] Compare each stage with the final BASE trajectory using the frozen Direct
  evaluator before training any weights.
- [x] Freeze necessary conditions before reading the result: all stage nets
  nonnegative, Z-stage net at least +12/128, at most four valid→invalid flips
  per stage, and mean stage net at least +12.
- [x] Terminal result: BASE `59/128`; lattice oracle `122/128` (net +63), X
  `102/128` (+43), Y `54/128` (-5), Z `43/128` (-16). Z causes 19
  valid→invalid and only 3 invalid→valid flips.
- [x] **No-go:** later MP20 coordinates are incompatible with already committed
  rollout errors. Do not submit the active-group CE training wrapper.
- [x] Canonical target re-audit remains no-go: lattice/X/Y/Z Direct is
  `122/95/57/43` versus BASE `59`; Y and Z are still non-adverse failures.

## P2 — paired active-group training

- [x] Closed without training after P1.5 failed the causal necessary condition.
- [x] Slurm151 remains an unexecuted engineering artifact; no checkpoint or
  scientific result was produced.
- [x] Do not reinterpret the negative realizability result as a hyperparameter,
  seed or update-count problem.

## P3 — untouched matched raw test

- [x] Not run because there is no causally admissible candidate checkpoint.
- [x] Preserve the untouched 128 Plans for a future method with an explicit
  geometry and stability mechanism.

## P4 — continuous periodic geometry interface design

- [x] Specify an identity-preserving interface; do not
  implement it in parallel with pilot science.
- [x] Inputs: DLM hidden states, q0 soft lattice/coordinates, species and site mask.
- [x] Lattice output: zero-initialized residual in SPD metric/Cholesky space.
- [x] Coordinate output: zero-initialized tangent residual on the fractional
  three-torus with exact triclinic PBC messages.
- [x] Convert continuous means and uncertainty to adjacent legal token logits;
  keep hard N/elements unchanged.
- [x] Loss: token CE + torus coordinate + metric + species-aware PBC distance;
  no inference CHGNet, best-of-N, reranking or completed-sample repair.
- [x] Specify O(N²) time, bounded-image memory, invariance tests, step0 equality,
  and a fixed-256 matched promotion contract.
- [ ] Implement only after the pilot result selects whether rollout matching is
  sufficient or the continuous executor is needed.

## P5 — canonical MP20 execution contract

- [x] Audit all teacher bodies against inference element prefill order.
- [x] Record train `4069/27136` and validation `1296/9047` exact-order matches;
  all rows preserve the correct species multiset.
- [x] Build full `27136/9047` canonical data by permuting complete
  species-coordinate site records to Plan order; drop zero rows.
- [x] Verify physical structure, lattice, body length, composition, source row
  and split are unchanged for every row.
- [x] Job 39282 trained one fresh Compact-V2 LoRA with the frozen two-epoch
  schedule and produced the sole step3392 checkpoint.
- [x] Verify job39282 pre-science state: checkpoint path null, world size2,
  canonical data path, fresh LoRA and exact step0 equality.
- [x] Verify training-log records through step1900 have finite
  loss/task loss/gradient norm/LR.
- [x] Monitor step1696 without selecting it; validation loss is `2.226879` and
  the frozen stage2 learning-rate restart is active.
- [x] Verify all subsequent records through the sole step3392 endpoint.
- [x] Job39321 trained the single canonical G2-PBC-R full epoch in `00:54:14`
  on two A800 and produced the sole step1696 checkpoint. Relation step0 delta
  is exactly zero; rank64/image-radius2/uncertainty-off config and all finite
  geometry/relation diagnostics passed. Validation loss is `2.475367`.
- [ ] Job39335 reuses the frozen prospective 256 Plans and old G2-PBC-R raw
  control, generating only canonical DLM and canonical DLM+G2 with identical
  stream17 noise before fast Direct. It does not resample the Planner or run
  model494/CHGNet.
- [x] Freeze the previously effective G2-A recipe for that epoch: acyclic
  `q0 -> soft geometry -> residual -> q1`, rank-64 two-layer relation adapter,
  exactly-zero output initialization, strict triclinic 125-image minimum image,
  and normalized species-aware overlap margin
  `clamp(0.55(r_i+r_j), 0.60 A, 1.40 A)`.
- [x] Freeze the proven geometry objective weights: metric `0.1`, pair/RDF
  `0.1`, overlap `0.2`, coordination `0.05`; geometry-only masking keeps N and
  element tokens visible.
- [x] Retain exact `7+4N`, hard N/elements, legal-family masks,
  duplicate-coordinate mask and lattice-volume mask at decoding.
- [x] Keep the detached uncertainty gate off: prior B/RU did not beat A/R and
  had adverse raw-energy direction. Do not revive Force/BTRD or SGTC.
- [ ] Matched raw evaluation treats Direct and CHGNet as co-primary, with
  stability prioritized; no checkpoint/seed/epoch selection.
- [ ] If Direct improves without raw-energy improvement, do not add CE epochs;
  move only to the preregistered MP20-only stability-conditioned periodic score
  executor described in `DLM_DIRECT_STABILITY_DECISION_20260902.md`.

## Resources and operations

- Maximum 6 A800, 4–8 CPU per GPU, at most 2 jobs.
- Use only the existing `tmux ssha800`; never reconnect `ssha800_2`.
- No `nvidia-smi` or unrelated CUDA probes.
- Normal intermediate heartbeats are quiet. Notify on rollout data, training,
  raw Direct, unrecoverable failure or SSH loss.

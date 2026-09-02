# Rollout-Matched DLM 24-hour checklist V1

Deadline: 2026-09-03 23:30 Asia/Shanghai.

## Current state (2026-09-02)

- Corrected immutable cohort:
  `rollout_matched_pilot_128x128_v2_20260902`.
- It contains 128 rollout-train and 128 untouched holdout rows, with zero exact
  composition overlap and deployment-mode `C3FD-predicted-native-V2` prompts.
- The earlier V1 cohort and capture job 39253 used teacher prompts. Job 39253
  was cancelled before training; both are retained only as engineering traces.
- Corrected exact-axis capture job 39259 is running on one A800. No pilot
  training or holdout result exists yet.
- The geometry execution interface is specified in
  `TOKEN_NATIVE_PBC_GEOMETRY_EXECUTOR_V1.md`; implementation waits for the
  rollout-matched pilot result.

## Objective

Improve raw periodic geometry and stability after C3FD has already solved
composition validity. Planner, composition distribution, dynamic `7+4N`,
exact-axis inference, temperature and model494 tau800 remain frozen.

The pilot tests whether training on the DLM's own committed exact-axis errors
improves continuation geometry. It is not another synthetic planned-corruption
run: historical B3 improved synthetic NLL but worsened actual-rollout NLL and
Strict S.U.N.

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
- [x] Fill masked suffix positions only for storage; record the original model
  mask and the active-group supervision mask separately.
- [ ] Job 39259 produces exactly `128 × 4 = 512` transition rows; no retries or
  replacement.

## P2 — paired active-group training

- [ ] Reuse the promoted G2-A LoRA and relation adapter.
- [ ] Model input: committed rollout tokens plus masks at current/future groups.
- [ ] CE supervision: current active group only.
- [ ] First pilot uses active-group CE only. Periodic auxiliary losses are zero
  so the test isolates rollout-state matching; future groups are input masks and
  never auxiliary targets.
- [ ] Jointly update LoRA + G2 for one seed, 128 updates, only step128 eligible.
- [ ] Assert prompt/N/element order equality, exact body length, finite losses,
  nonzero LoRA and relation gradients, and two-GPU synchronization.

## P3 — untouched matched raw test

- [ ] Generate BASE and candidate on the 128 holdout Plans with identical noise.
- [ ] Compute body, composition validity and fast Direct only.
- [ ] Promotion requires body and composition within 1 percentage point,
  Direct at least `+4/128`, and net invalid→valid at least `+4`.
- [ ] If the gate fails, stop before CHGNet/model494 and archive the negative.
- [ ] If it passes, run paired raw CHGNet; require non-adverse median energy
  before model494/S.U.N.

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

## Resources and operations

- Maximum 6 A800, 4–8 CPU per GPU, at most 2 jobs.
- Use only the existing `tmux ssha800`; never reconnect `ssha800_2`.
- No `nvidia-smi` or unrelated CUDA probes.
- Normal intermediate heartbeats are quiet. Notify on rollout data, training,
  raw Direct, unrecoverable failure or SSH loss.

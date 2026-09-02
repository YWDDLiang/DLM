# Force-Score periodic DLM checklist V1

## Objective

Improve raw structural validity and stability without changing the Planner,
composition distribution, `7+4N` representation or model494 tau800 endpoint.

The proposed transition is:

\[
L=L_{\mathrm{CE}}+L_{\mathrm{G2-valid}}+L_{\mathrm{ForceScore}},
\]

where CE preserves the exact crystal language, G2-valid defines the analytic
periodic feasible region, and Force-Score supervises the local energy-descending
transition carried by the existing G2 residual.

## Current-method failure audit

- Scope: current promoted G2 prospective/raw-refined cells, full-epoch A
  raw/refined cells, and current Plan1200 accounting only.
- Excluded: BASE, G0/G1, native canary, uncertainty-gated B and retired methods.
- Current result: 1,024 structural observations, 260 Direct structural failures,
  all 260 explained by periodic minimum distance below 0.5 Å; volume failures 0.
- Additional interface failures: 10 upstream parse/body failures in the four G2
  cells and 20 Plan1200 CIF construction failures.

## Phase A — 512-row teacher preflight

This is a mechanism test, not a training dataset or paper result.

- 64 independent MP20-train structures × 8 perturbations = 512 rows.
- Distribution: 256 collision, 128 coordinate jitter, 64 near-threshold, and 64
  lattice/wrap sentinels.
- No energy, hull, test or prospective outcome is read during selection.
- Measure CHGNet force/stress coverage, one-step energy change, periodic minimum
  distance, invalid→valid transitions, hard-token retention and sub-bin rate.
- Compare continuous force targets with the same targets quantized back through
  the exact dynamic `7+4N` representation.

Phase A supports proceeding only if the force direction usually lowers CHGNet
energy, does not damage valid structures, improves collision geometry, and its
effect survives token quantization often enough to supervise q1.

## Phase B — micro-student preflight

Only after Phase A succeeds:

- split by base structure: 48 train / 16 holdout;
- train G2 residual only for 64–128 updates;
- freeze Planner, base DLM, Compact-V2 LoRA and q0;
- use wrapped soft XYZ targets plus a secondary direction loss;
- verify holdout collision, raw Direct, paired CHGNet and gradient compatibility.

This phase tests whether a valid teacher can actually be learned through the G2
residual. It is not the final model.

## Phase C — full MP20 training

Only after both preflights succeed:

- complete MP20 train: 27,136 independent structures;
- one deterministic perturbation/current state per structure, weighted to the
  audited current-method failure distribution;
- one training seed, effective batch 16, one epoch ≈ 1,696 updates;
- release only the epoch endpoint; no checkpoint or hyperparameter selection;
- full MP20 validation is evaluation-only;
- final screen reuses the fixed current 256 Plans and noise, raw first;
- run model494 tau800 and official S.U.N. only after a positive raw result.

## Fixed boundaries

- no Planner resampling or composition tilt;
- no inference-time CHGNet, force calculation, repair or candidate selection;
- parser/schema failures stay under CE and exact-token accounting;
- severe non-graphable geometry stays under G2-valid and remains in denominators;
- official hull is never a training target.

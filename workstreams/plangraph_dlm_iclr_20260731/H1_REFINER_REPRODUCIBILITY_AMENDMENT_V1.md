# H1 Continuous-Refiner Reproducibility Amendment V1

State: `active_after_retained_engineering_stop`

Date: 2026-08-02

This amendment does not accept or overwrite failed job 29646. It corrects an
invalid reproducibility assumption discovered by that zero-change control
replay.

## Retained evidence

- Frozen H1 P0 reference: run
  `20260731_h1a2c_p0_p1_sun256_exploratory_v1`, P0 generation job 28983.
- Zero-change replay: run `20260802_h1_exact_replay_p0_v1`, job 29646,
  `FAILED 1:0` because the registered byte-parity gate failed.
- Both executions used node99, one A800, the same frozen P0 attempt ledger,
  B0 checkpoint, body batch 8, parent batch 64, `model_494`, paired noise
  identities, and exactly 800 reverse steps.
- `body_attempts.jsonl` is byte-identical:
  `d1970e1499e8837d1b837a085005fdcbbc24c9ca12e2999d2f0e3c36d015816f`.
- All 246 proposal-graph records are semantically exact: metadata, tensor
  shapes, dtypes, and tensor values have zero mismatches.
- All 256 generation metadata/status records match; the same 246 ordinals
  succeed and the same 10 fail, with identical species and site counts.
- The 246 continuous structures are not byte-identical. Periodic fractional
  coordinate absolute difference has median 0.248 and maximum 0.500; relative
  volume difference has median 1.33%, p95 12.49%, and maximum 23.65%.

## Cause

The frozen CrysLLMGen refiner uses CUDA `torch_scatter` reductions. Their
floating-point reduction order is not bitwise deterministic. The exact same
proposal tensors and stateless noise therefore begin with small numerical
differences, which the iterative 800-step diffusion trajectory amplifies.
Byte equality of final floating-point coordinates is consequently not a valid
control gate, even on the same A800 node.

## Amended gates

The failed byte-parity result remains an engineering stop and is never relabelled
as a pass.

Exact equality remains mandatory for:

1. Planner/body ordinals and all sampling-noise identities;
2. raw and canonical Plan identities;
3. `body_attempts.jsonl`;
4. proposal-graph metadata and every tensor;
5. body/refiner checkpoints, tokenizer, batching, source, and 800-step config;
6. success/failure denominator, species, and site counts.

Continuous refined coordinates are evaluated statistically:

- no direct-metric or S.U.N. claim may rely on one historical coordinate
  realization versus one fresh realization;
- before such a claim, freeze a common refiner-repeat ledger and run the H1
  control and the single-factor candidate under the same repeats;
- report every repeat, the pooled raw all-attempt result, paired bootstrap or
  McNemar intervals as appropriate, and whether the effect sign is stable;
- the repeat count and pass threshold must be fixed before candidate metrics
  are inspected.

## Scope firewall

The amendment changes evaluation of continuous numerical reproducibility only.
It does not change a model, checkpoint, Plan, body schedule, seed identity,
denominator, S.U.N. cache, or scientific threshold. Body-only engineering
stages R1/R2 may proceed without the refiner. Any refined direct/S.U.N. stage
remains blocked until the repeat ledger is preregistered.

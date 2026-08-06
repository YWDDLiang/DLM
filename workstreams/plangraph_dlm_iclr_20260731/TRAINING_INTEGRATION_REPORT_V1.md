# PlanGraph-DLM Training Integration Report V1

Date: 2026-07-31

Status: `remote_full_denominator_preflight_passed_engineering32_manifest_frozen_no_submission`

## Outcome

The planned-corruption method is integrated into the existing LLaDA SFT path
without changing the continuous refiner or the frozen H1 artifacts. The
registered primary ratio is `iid:planned = 2:1`; D1 uses current generation
order and D2 uses validated PlanGraph dependencies.

This report is implementation evidence only. It is not a G1/G2 scientific
result and does not authorize a job submission.

## Enforced behavior

- `planned_fraction=0` takes the legacy Torch-RNG D0 branch with the original
  random-call order.
- Planned examples keep prerequisite groups visible, mask all future groups in
  the input, and supervise only a stochastic subset of the active group.
- Every record has a content-derived stateless corruption key; row order,
  sample IDs, evaluation metadata, and energy fields do not enter the key.
- Validation uses one fixed stateless-iid mask panel across checkpoints and
  arms.
- D2 requires a validated PlanGraph body record and refuses CSV-only fallback.
- D2-shuffle keeps the exact D2 rows and position groups, but changes only the
  dependency order using a content-keyed SHA-256 rank and frozen seed.
- Planned corruption refuses truncation and refuses any tokenizer that does
  not map each dynamic semantic token to exactly one tokenizer token.

## Data publication gate

The new dataset builder:

- retains every non-empty source row in the denominator and original order;
- emits separate Planner and dynamic-body JSONL views;
- copies no source metadata or sample identifier;
- records ordered per-split `training_pair_sha256` ledgers;
- rejects any conversion failure and any train/validation pair overlap;
- refuses an existing output path and every frozen H1 output root; and
- publishes atomically with file hashes, `manifest.json`, and `_SUCCESS`.

The builder passed synthetic atomicity, leakage, denominator, overlap, and
H1-guard tests, then published the frozen remote dataset at
`data/dlm_sft/mp_20_plangraph_v1_20260731`. All 36,183 rows converted, the
eight registered output hashes were reverified after publication, and there
was no cross-split training-pair overlap. The dataset manifest SHA-256 is
`2aac3b91a9cc5f0b18bfa0084886f7cb7e5eca689d0fdce3235f41ccfc0b2e2e`.

## Tokenizer and mask preflight

The preflight CLI verifies the published dataset hash ledger, registers the
ordered data vocabulary exactly as training does, audits every requested row,
hashes the exact ordered schedule, and runs bounded CPU smoke tests for:

1. planned-only D1, D2, or D2-shuffle;
2. the registered 2:1 iid:planned mixture; and
3. fixed stateless-iid validation.

The smoke test checks nonzero supervision, visible prerequisites, fully masked
future groups, active-group-only loss, prompt preservation, iid input/loss
equality, and deterministic replay.

## Verification evidence

- Relevant local regression suite: 101 passed, 0 failed.
- Frozen remote source bundle V2: 34 passed, 0 failed.
- Pure modules and entry points compile successfully.
- Ruff passes on the new builder, preflight, and tests.
- CPU/CUDA stateless masks match exactly on the local NVIDIA GeForce RTX 5060.
- The H1 verifier reports six manifests and 68/68 registered entries intact.
- D1, D2, and D2-shuffle each passed the real-tokenizer preflight on
  36,183/36,183 rows at `max_length=768`, with no failed row, duplicate
  corruption key, or mask-invariant failure.
- The observed maximum model lengths were 750 on train and 696 on validation.
  A retained `max_length=382` D1 report failed only because the strict
  no-truncation guard correctly rejected 30,956 rows; no training was started
  from that invalid extrapolation.
- D2 and D2-shuffle have identical group-count distributions and different
  ordered schedule hashes, confirming that the mechanism control changes only
  dependency order.

The local CUDA device is not the registered A800 and cannot replace the
32-attempt A800 engineering pilot.

## Execution boundary

The canonical H1 fallback still verifies 68/68 locally. The A800 worktree is an
incomplete, non-authoritative mirror: two top-level checksum manifests and 19
`baseline/` files are absent, while every present registered file matches its
expected SHA-256. That mirror must not be patched or used as a restore source.
No H1 path was changed.

`ENGINEERING_PILOT_32_MANIFEST_V1.json` freezes a four-arm D0/D1/D2/D2-shuffle
runtime pilot over 32 train microbatches and four optimizer updates per arm.
It measures memory, wall time, finite loss/gradient behavior, and scheduling
invariants only. It does not authorize Slurm submission, checkpoint selection,
generation, refinement, S.U.N., or any automatic downstream action. No model
training or Slurm submission has occurred.

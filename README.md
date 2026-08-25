# H1-A2 crystal generation

This repository contains the paper-facing H1-A2 training and inference
workflow:

```text
learned rich-Plan source
    -> composition/N-anchored masked geometry completion
    -> identity-preserving continuous diffusion refinement
```

All project paths are resolved relative to the repository root. Scientific
sampling settings are defined in code. The coordinate-grouped decoding option
is exposed as a single inference switch.

## Reported result

| Method | Entries | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|
| CrysLLMGen reference | 1,000 | 90/1000 = 9.00% | 461/1000 = 46.10% |
| **H1-A2 method family — future paper S.U.N. main table** | **1,000** | **105/1000 = 10.50%** | **488/1000 = 48.80%** |

The `105/1000` Strict and `488/1000` Meta pair remains the planned paper
S.U.N. main-table value. Cohort-level evidence views are reported separately:
the exact all-requested-attempt audit is `103/1200` Strict and `553/1200`
Meta, while the historical frozen compatibility view is `94/1000` and
`474/1000`. These views are not silently substituted for one another. The
repository also provides a frozen-Plan `256 x 4` downstream control.

## Entry points

```bash
# Inspect placeholders before the A800 assets are published.
sbatch slurm/00_environment.sbatch

# Submit the full H1-A2 chain. Missing DLM/refiner checkpoints default to
# training and successful training jobs automatically feed inference.
bash scripts/submit_h1a2.sh

# Replay the frozen 256-Plan quick route four times. Grouped coordinate
# decoding is enabled by default for this route.
bash scripts/submit_quick_256x4.sh

# If a Planner checkpoint is present, regenerate 256 Plans with the fixed
# Planner sampling seed instead of replaying the frozen Plan file.
RESAMPLE_PLANS=true bash scripts/submit_quick_256x4.sh
```

The full route is fully de novo only when it samples Plans from the learned
Planner. The frozen `256 x 4` route is intentionally a Plan-conditioned
body/refiner control; it reproduces downstream behavior but is not used to
claim de novo Plan generation.

See [REPRODUCTION.md](REPRODUCTION.md) for environment, data, training,
inference, evaluation, and resume behavior.

Construction progress and A800 handoff items are tracked in
[`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

Reviewer-safe problem and contribution framing is developed in
[`docs/PAPER_POSITIONING.md`](docs/PAPER_POSITIONING.md).
The disabled candidate methods and their promotion boundary are recorded in
[`docs/CANDIDATE_STATUS.md`](docs/CANDIDATE_STATUS.md).
The mapping from the proposal--realization story to required analyses and
minimum additional inference is in
[`docs/PROPOSAL_REALIZATION_EVIDENCE.md`](docs/PROPOSAL_REALIZATION_EVIDENCE.md).
The top-conference related-work map and claim boundaries are in
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).
The distinction between learned de novo Plans and replay/control Plans is in
[`docs/DE_NOVO_SCOPE.md`](docs/DE_NOVO_SCOPE.md).
The concept-only ICLR review is recorded in
[`docs/STORY_REVIEW.md`](docs/STORY_REVIEW.md).

The remaining evidence and release gaps are intentionally listed only at a
high level in [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

## Current asset status

The environment lock, model checkpoints, full MP-20 split, and frozen R03
Plan/seed ledger are documented placeholders. They will be populated from the
frozen A800 workspace before release. No empty model file is used as a fake
checkpoint.

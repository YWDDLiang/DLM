# Next execution: Plan + DLM null-repair gate

Status: prepared locally; not installed remotely; not submitted.

## Why this is the next run

The old S2 exact-null failure was a trajectory-replay failure, not evidence that
the trained adapter emitted a nonzero null delta.  `force_null` already calls
the frozen parent denoiser directly.  Replaying parent and null independently
on CUDA allows nondeterministic scatter reductions to diverge and the
800-step reverse loop amplifies the difference.

The repaired gate therefore gives R0 and R1 two semantic labels over one
reverse execution and one result object.  Matched and shuffled Plans remain
separate paired trajectories.

## Frozen run

- identity: `plan_dlm_null_repair_v1`
- data: 256 eligible frozen R5-C drafts
- overlap with the earlier observed gold-gate panel: 0
- arms:
  - R0 frozen parent
  - R1 exact-null alias of R0
  - R2 matched PlanLite
  - R3 deterministic within-stratum shuffled PlanLite
- environment:
  `/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python`
- resources: ordinary GPU partition, 1 A800, 4 CPU, 64 GiB, 8 hours
- outputs: structures, CrysLLMGen direct metrics and one null-repair decision
- excluded: training, S.U.N., CHGNet, MP API, MLIP, DFT, retries and reranking

## Remote sequence

1. Verify the final execution archive SHA256:
   `ab6638b5d06047a9eb3aae6f65a6bc4c4a2bd86905469332b6f51f8261e22d0a`.
2. Extract it into a fresh staging directory.
3. Verify source-manifest SHA256:
   `77d90b6743a91894886d9db5b3c9b4252f57651b3afb20800d97f885e9167122`.
4. Run `install.py` against the shared-Plan project.  The installer fails
   closed on unexpected pre-existing source and writes one installation
   record.
5. In `diff_meets_diff`, run the focused server tests, including the
   torch/torch-scatter exact-null boundary checks.
6. Submit `scripts/server/submit_plan_dlm_null_repair.sh` once.  It does not
   inspect or interfere with unrelated queues.
7. Monitor only state, stage, stderr anomalies and the terminal decision.

## Promotion boundary

Pass requires:

- R0/R1 coordinate and lattice difference exactly zero;
- one parent/null reverse execution per batch;
- object alias true and independent CUDA replay false;
- matched-minus-shuffled lattice-family hit at least 5 percentage points;
- CrysLLMGen validity/coverage noninferiority within the frozen margins.

If it passes, the next scientific panel replaces gold PlanLite with the frozen
H1-A2 epoch-2 Planner and frozen R5-C exact-length body, then evaluates
CrysLLMGen direct metrics and the original A100-script/CHGNet S.U.N. protocol.
That later panel is the first fully de-novo result on the relaunched ICLR
mainline.

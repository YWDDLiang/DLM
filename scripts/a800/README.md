# Active A800 Utilities

The previous R3--R5, DN, H1/H2, free-geometry, A100, and cleanup launchers were
archived on 2026-07-10. This directory intentionally contains only reusable
cluster infrastructure:

- `bootstrap_diff_meets_diff.sh`
- `check_ssha800.sh` (read-only; never reconnects SSH)
- `env_doctor.py`
- `slurm_submit.sh`
- `slurm_cpu_submit.sh` (four-job-capped CPU preprocessing/audit lane)
- `run_p1_shard.sh` and `audit_p1_dataset.sh` (eight immutable shards per split)
- `run_day7_lane.py` (four-lane, modulo-partitioned execution of the immutable
  materialized Day-7 registry; no shell interpolation or completed-cell reuse)
- `reconcile_day7_terminal_timeout.py` (accounting-only final-cell recovery;
  requires immutable Slurm `TIMEOUT` evidence, appends missing attempts as
  terminal timeouts, and never invokes the model or retries an attempt)
- `benchmark_wqcodiff_v39_primitives.py` (thread-contract-enforced local/server
  microbenchmark for exact event, geometry, and D3PM fast paths)
- `compare_recovery_equivalence.py` (attempt-level v38/v39 semantic gate;
  compares immutable artifacts without editing or retrying either run)
- `run_week2_training_job.py` (one immutable shared/screen training job from
  the post-Day-7 plan, with dependency optimizer-checkpoint hashes)
- `run_week2_sampling_lane.py` (four-lane matched 256-preflight and
  `3x1000` development sampling, including the frozen DISC-ONCE tau grid)
- `prepare_mlip_assets.py` (checkpoint-derived element support, hashes, and
  immutable three-MLIP asset lock)
- `mlip_runtime_probe.py` (one-evaluator-per-process CPU/CUDA checkpoint smoke)
- `build_source_bundle.py` and `install_source_bundle.py` (one-time,
  source-only, manifest-verified deployment)

New launchers must:

1. use an experiment ID from the WQ-CoDiff flagship plan;
2. load `configs/experiments/wyckoff_codiffusion/protocol_v3.yaml` and reject v1/v2;
3. emit stable `attempt_id`, seed, data/model hash, failure status, and timing;
4. never select candidates based on rank order, atom count, validity, or output;
5. write into a new run directory rather than reuse a legacy path.
6. export the registered BLAS thread limits and offline model flags;
7. default to one A800 per model job and never allocate over four GPUs total.
8. set `WQ_RUNTIME_SCOPE` to `chgnet`, `mattersim`, or `mace` for evaluator
   jobs; training uses `core`. MatterSim's isolated `PYTHONPATH` is injected
   only by the registered Slurm launcher.

Sampling and corruption recovery use ragged batched inference (registered
default 64). Changing the batch size may address memory pressure but must not
change attempt IDs, derived seeds, call counts, or output accounting.

Historical scripts are available under
`archive/20260710_pre_wyckoff/legacy_scripts/a800/` for provenance only.

The active offline evaluator stack is `wqcodiff-evaluator-stack-v4` and binds
only `wheelhouse_v4/`, `source_sdists_v4/`, and
`wheelhouse_lock_v4.json`. The unversioned and v2 wheelhouse locks, together
with the failed v3 run, are preserved as failure evidence and must not be
modified or used by a runtime. MatterSim's v4 isolated target additionally
provides the locked ASE 3.27.0 and setuptools 81.0.0 compatibility APIs without
changing core packages, ABI packages, Triton, or CUDA packages.

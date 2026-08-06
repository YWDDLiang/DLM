# Four-Week WQ Co-Diffusion Execution Runbook

This file is the operational entry point. Scientific choices live in
`protocol_v3.yaml`; job-cell choices live in `experiment_registry_v1.yaml`.
Their SHA256 values are recorded before the first checkpoint.

## Non-Negotiable Remote Rules

- Local files are the sole code source. The server is an execution copy and
  never uses Git.
- Remote commands run only in the existing user-owned `ssha800` tmux SSH pane.
- If that pane is no longer connected, stop immediately and ask the user. Do
  not reconnect SSH.
- Exactly one source-only SCP upload is allowed after local tests pass.
- Preserve remote `data/`, `reference/`, `runs/`, `reports/`, `.secrets/`, and
  model assets. Never use delete-sync.
- Login-node network is used only for the pinned wheel/model download. CUDA
  work runs only through Slurm; GPU jobs are offline.

## Day 1 Source and Environment Gate

Build the deterministic local bundle:

```bash
python3 scripts/a800/build_source_bundle.py \
  --output-dir source_bundles/20260717_frozen
```

After the existing tmux connection is verified, upload only the `.tar.gz` by
the single authorized SCP. In the existing remote pane, extract the installer
to staging and run it with the printed bundle SHA256. The installer validates
every member and overlays only the source whitelist.

On the login node, first run the non-mutating dependency resolver. The explicit
CHGNet metadata waiver retains Torch 2.4.0 and is limited to the single
`chgnet==0.4.2` declaration of `torch>=2.4.1`:

```bash
RUN_ID=week1-env-dry APPLY=0 ALLOW_CHGNET_TORCH_WAIVER=1 \
  ALLOW_MATTERSIM_INFERENCE_RUNTIME=1 \
  bash scripts/a800/bootstrap_diff_meets_diff.sh
```

Stop if the resolver would replace Python 3.10, Torch 2.4.0, NumPy/materials
ABI packages, Triton, or CUDA packages. Only after that report passes:

```bash
RUN_ID=week1-env-apply APPLY=1 ALLOW_CHGNET_TORCH_WAIVER=1 \
  ALLOW_MATTERSIM_INFERENCE_RUNTIME=1 \
  bash scripts/a800/bootstrap_diff_meets_diff.sh
python scripts/a800/prepare_mlip_assets.py --download --smoke
```

This creates the active immutable `wheelhouse_v4/`,
`wheelhouse_lock_v4.json`, and MLIP locks under
`/public/home/jiaosz/ywliang/models/wqcodiff/`. The earlier
`wheelhouse_lock.json` and `wheelhouse_lock_v2.json` are immutable failed-attempt
evidence and must never be edited, deleted, or used by a runtime. A missing
required asset is a hard stop.
Element support is read from each frozen checkpoint/model object and
recorded with a `support_basis`; it is never inferred from the MP20 vocabulary.
The dependency waiver is also immutable and bound by SHA256 into the MLIP asset
lock. The only source-built wheels are the exact pure-Python
`nvidia-ml-py3==7.352.0` and `python-hostlist==2.3.0` sdists; both source and
built-wheel hashes are locked. Those wheels are built once with pip's wheel
cache disabled, `SOURCE_DATE_EPOCH=315532800`, and `PYTHONHASHSEED=0`; once the
v4 lock exists, later runs verify rather than rebuild them. The pre-run global
`pip check` output is also
hash locked because the shared environment contains unrelated legacy-package
failures that cannot be repaired without changing protected NumPy/pymatgen.
Those lines must remain exactly unchanged; within the project closure, any
mismatch other than the registered CHGNet/Torch line is a hard failure.

MatterSim and MACE have incompatible `e3nn` requirements. The environment
bootstrap therefore keeps MACE/CHGNet in the core conda environment and creates
an immutable MatterSim inference target at
`models/wqcodiff/runtimes/mattersim-1.1.2-py310-v4`. Its reviewed import closure,
resolver report, official wheel, and complete file tree are hash locked. Do not
manually add that target globally. Every evaluator job declares exactly one
scope:

```bash
WQ_RUNTIME_SCOPE=chgnet    bash scripts/a800/slurm_submit.sh RUN_ID COMMAND...
WQ_RUNTIME_SCOPE=mattersim bash scripts/a800/slurm_submit.sh RUN_ID COMMAND...
WQ_RUNTIME_SCOPE=mace      bash scripts/a800/slurm_submit.sh RUN_ID COMMAND...
```

Training/sampling jobs use the default `WQ_RUNTIME_SCOPE=core`. The Slurm
launcher adds the MatterSim target only for `mattersim`; loading multiple MLIPs
in one process is not paper eligible.

MatterSim's metadata does not protect it from newer packaging APIs. Its
isolated runtime therefore additionally pins official `ase==3.27.0` and
`setuptools==81.0.0` wheels; their SHA256 values and the successful full import
closure are part of the v4 lock. These pins never replace core packages.

## P0/P1/P2 Gate

Preprocess all three MP20 splits as eight immutable shards per split, retaining
every failed record. Each CPU job processes the same numbered shard of train,
validation, and test. Submit shards 0--3 first and, after they finish, shards
4--7 so that no more than four project CPU jobs are active:

```bash
for index in 0 1 2 3; do
  WQ_STAGE=p1-preprocess JOB_NAME="wqcpu-p1-${index}" \
    bash scripts/a800/slurm_cpu_submit.sh "week1-p1-${index}" \
    bash scripts/a800/run_p1_shard.sh "${index}"
done

# Run this second wave only after the first wave has reached terminal states.
for index in 4 5 6 7; do
  WQ_STAGE=p1-preprocess JOB_NAME="wqcpu-p1-${index}" \
    bash scripts/a800/slurm_cpu_submit.sh "week1-p1-${index}" \
    bash scripts/a800/run_p1_shard.sh "${index}"
done

WQ_STAGE=p1-audit JOB_NAME=wqcpu-p1-audit \
  bash scripts/a800/slurm_cpu_submit.sh week1-p1-audit \
  bash scripts/a800/audit_p1_dataset.sh \
  runs/week1-p1-audit/outputs/p1_dataset_audit.json
```

Combine paths only through the audit; never concatenate and rewrite the source
records. Run the P2 numerical gates as registered CPU jobs as well:

```bash
WQ_STAGE=p2-formal JOB_NAME=wqcpu-p2-formal \
  bash scripts/a800/slurm_cpu_submit.sh week1-p2-formal \
  python -m crystal_dlm.wqcodiff formal-audit --transitions 1000000 \
  --output runs/week1-p2-formal/outputs/formal_audit.json

WQ_STAGE=p2-chart JOB_NAME=wqcpu-p2-chart \
  bash scripts/a800/slurm_cpu_submit.sh week1-p2-chart \
  python -m crystal_dlm.wqcodiff chart-audit \
  --output runs/week1-p2-chart/outputs/pyxtal_chart_audit.json
```

The full P1 audit enforces exactly 45,229 records, per-split coverage at least
95%, round-trip at least 99%, atom-count consistency 100%, and zero
canonicalized cross-split leakage. It also reports selection-induced material
family shift without hiding failed decompositions. The exhaustive chart audit
checks every Wyckoff position in all 230 space groups using PyXtal's pinned
spglib Hall style and the exact runtime tangent projector; both errors must be
strictly below `1e-6`.

Create the Day-7 training subset by material-ID hash only:

```bash
train_parts=()
for index in $(seq 0 7); do
  train_parts+=(--dataset \
    "data/wqcodiff/p1_v3/train.part-$(printf '%03d' "${index}")-of-008.jsonl")
done
python -m crystal_dlm.wqcodiff dataset-subset "${train_parts[@]}" \
  --fraction 0.10 --salt day7-train-v1 \
  --output runs/week1-day7/outputs/train_10pct.jsonl
```

Run one Slurm CUDA/model smoke and one three-MLIP smoke before training. Every
submission sets `WQ_WEEK=1`, explicit `WQ_METHOD`/`WQ_STAGE`, and
`PROPOSED_GPU_HOURS`, uses one GPU lane, and first passes
`scripts/a800/env_doctor.py`.

## Day 4–7: DLM Falsification and Threshold Freeze

For the 10k development schedule, train one shared WQ checkpoint to update
6,000 and fork the five registered variants from that exact model, optimizer,
scheduler, EMA, sampler, and RNG state. Never train five independent shared
prefixes.

Materialize the registered job cells:

```bash
python -m crystal_dlm.wqcodiff experiment-plan \
  --run-id week1-day7 \
  --source-bundle-sha256 SOURCE_BUNDLE_SHA256 \
  --output runs/week1-day7/notes/day7_job_plan.json
```

First execute only the 45 threshold-calibration cells. Freeze the result:

```bash
python -m crystal_dlm.wqcodiff revision-calibrate \
  --input CALIBRATION_ARTIFACT.jsonl \
  --output runs/week1-day7/notes/revision_threshold_lock.json
```

Inject the frozen threshold into the remaining plan; no default value is
allowed. Then run 180 primary and 72 intervention cells. Aggregate with 10,000
paired bootstrap repetitions. A failed DLM gate deletes the DLM-superiority
claim and selects the better AR/D3PM engine. Sampling and recovery default to
ragged `--inference-batch-size 64`; a smaller registered value is allowed only
for memory, never to alter the attempt set or seeds.

## Day 8–14: Matched Screening

Train one full-data WQ shared checkpoint and one atom-representation shared
checkpoint to update 60,000 on the registered 100k cosine schedule. Each route
forks from its matching shared checkpoint and stops at update 85,000:

```bash
train_parts=()
for index in $(seq 0 7); do
  train_parts+=(--dataset \
    "data/wqcodiff/p1_v3/train.part-$(printf '%03d' "${index}")-of-008.jsonl")
done
python -m crystal_dlm.wqcodiff train \
  "${train_parts[@]}" --variant ROUTE --training-seed 11 \
  --updates 100000 --shared-checkpoint SHARED_60000.pt \
  --stop-after-update 85000 --output ROUTE_SCREEN_DIR
```

The 85k EMA is non-paper and used only for the registered 256-attempt smoke and
3x1,000 development screen. Selected routes resume the full checkpoint—not the
EMA—from 85k to 100k. Freeze the strongest matched WQ comparator and the
topology-preserving 16-call common refiner on Day 14.

## Day 15–21: Core Mechanism

Run fixed topology, birth/death-only, confidence, geometry, random-count,
shuffled-geometry, no-revision, and extra-call controls with matched pair IDs
and initial noise. Paper-eligible revision sampling must provide the Day-7
threshold lock. Stop the flagship mechanism if MatterSim `MLIP-SUN@0.1` gain is
below 2 pp, any sampling-seed direction is non-positive, or Novel&Unique drops
more than 2 pp.

P6 is allowed only after a Day-17 P5 pass. CHGNet may guide its 10k updates;
MatterSim and MACE are each used once for the frozen transfer gate. Freeze the
final method and all configuration on Day 21.

## Day 22–28: Frozen Confirmation

Train only missing seeds 23 and 47 for champion and final. Allocate attempts
`3334/3333/3333`, exactly 10,000 per method. The method-independent pair-ID hash
selects the same 6,000 attempts for all three MLIPs. Evaluate raw,
common-refiner, and relaxed stages with evaluator-specific caches and frozen
hulls. Run the final 10,000-repetition hierarchical bootstrap and workflow
audit.

Before every submission wave, audit consumed plus proposed GPU-hours:

```bash
python scripts/a800/gpu_budget.py \
  --runs-root runs --current-week WEEK --proposed-gpu-hours HOURS
```

Week cumulative ceilings are 180, 750, 1,250, and 2,050 GPUh. Before Week 4,
the audit also preserves at least 800 GPUh for frozen confirmation.

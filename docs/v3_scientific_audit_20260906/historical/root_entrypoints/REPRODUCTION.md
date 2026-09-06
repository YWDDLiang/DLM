# H1-A2 reproduction guide

This guide is the single public entry point for environment setup, data
preparation, training, inference, quick reproduction, and evaluation.

## 1. Environment

The historical run used Python 3.10 on NVIDIA A800 GPUs. The exact environment
export is pending confirmation on the A800 system. Until the files under
`environment/` are populated, do not substitute guessed package versions.

```bash
bash scripts/create_environment.sh
sbatch slurm/00_environment.sbatch
```

## 2. Data

The release data layout is:

```text
data/mp20/train.csv        # 27,136 rows
data/mp20/val.csv          # 9,047 rows
data/mp20/test.csv         # 9,046 rows
data/plans/r03_raw_256.jsonl
data/plans/r03_parsed_256.jsonl
data/plans/r03_seed_ledger_256.jsonl
```

The full frozen split will be tracked with Git LFS. A deterministic download
and preprocessing route is also provided:

```bash
bash scripts/download_data.sh
python scripts/prepare_data.py
```

At the current construction stage these files are placeholders documented in
`docs/PLACEHOLDER_ASSETS.md`.

## 3. Checkpoints

Expected relative destinations are:

```text
checkpoints/planner/
checkpoints/dlm/
checkpoints/diffusion/model_494.pt
```

`CHECKPOINT_ACTION=train` is the default for missing DLM or diffusion
checkpoints. Set `CHECKPOINT_ACTION=download` to use the future release
downloads. A missing Planner checkpoint does not block the frozen-Plan quick
route: the launcher falls back to `data/plans/r03_parsed_256.jsonl`.

## 4. Training

The three paper components have separate Slurm entry points:

```bash
sbatch slurm/10_train_planner.sbatch
sbatch slurm/20_train_dlm.sbatch
sbatch slurm/25_train_diffusion.sbatch
```

Known historical seeds are defined in `src/h1a2_repro/science.py`. Training
seeds that were not preserved in local evidence remain explicit placeholders
until the A800 audit is complete; the launchers will explain the missing item
instead of silently choosing a value.

## 5. Full H1-A2 inference

The full route requests 1,200 Planner attempts. Body generation consumes the
parsed Plans, and the first 1,000 body successes enter the 800-step continuous
diffusion refinement and evaluation stages.

```bash
bash scripts/submit_h1a2.sh
```

Standard H1-A2 decoding is the default. To enable grouped coordinate decoding
for a custom public run, set `SAFE_AXIS=true` before submission.

## 6. Frozen-Plan quick reproduction

The quick route replays the same 256 parsed Plans and scientific seed ledger
in four independent CUDA process realizations. It does not treat the four
processes as four independent Planner samples.

```bash
bash scripts/submit_quick_256x4.sh
```

Defaults:

```text
Plans: data/plans/r03_parsed_256.jsonl
Repeats: 4
Safe-axis: enabled
Refinement: model_494, 800 reverse steps, effective batch size 1
```

When `RESAMPLE_PLANS=true`, a present Planner checkpoint is used to sample 256
Plans with the fixed Planner seed. If the checkpoint is absent, the launcher
prints the fallback reason and uses the frozen Plan file.

## 7. Evaluation

The evaluation chain reports Direct composition/structure/joint validity,
coverage, novelty, uniqueness, CHGNet relaxation, and Strict/Meta S.U.N.
Materials Project credentials are read only from `MP_API_KEY`.

If `MP_API_KEY` is absent, all completed upstream outputs are retained and the
workflow stops before S.U.N. The user can export the key and resume only the
evaluation stage.

## 8. Outputs

All outputs are written under relative paths:

```text
runs/h1a2_full/
runs/quick_256x4/repeat_0/
runs/quick_256x4/repeat_1/
runs/quick_256x4/repeat_2/
runs/quick_256x4/repeat_3/
```

The launchers use readable errors for missing assets and environment
requirements. Runtime workflows do not use Python `assert` statements or
historical file-hash gates.


#!/usr/bin/env bash
#SBATCH --job-name=r03-registered-trial
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=200G
#SBATCH --time=04:30:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
RUNROOT="${R03_RUN_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" "$SOURCE/operations/r03_c3fd_main_20260907/run_trial.py" --run-root "$RUNROOT" --stage "${R03_STAGE:?}"

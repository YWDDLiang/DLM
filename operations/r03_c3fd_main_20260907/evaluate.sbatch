#!/usr/bin/env bash
#SBATCH --job-name=r03-registered-cpu-evaluation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:20:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
RUNROOT="${R03_RUN_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
export PYTHONPATH="$SOURCE/src" CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
phase=canary
if [[ "${R03_STAGE:?}" == pilot_evaluate ]]; then phase=pilot256; fi
if [[ "$R03_STAGE" == formal_evaluate ]]; then phase=formal; fi
"$PY" "$SOURCE/operations/r03_c3fd_main_20260907/run_evaluation_stage.py" --run-root "$RUNROOT" --phase "$phase"
if [[ "$R03_STAGE" == pilot_evaluate && ! -f "$RUNROOT/evaluation_canary/_SUCCESS" ]]; then
  "$PY" "$SOURCE/operations/r03_c3fd_main_20260907/run_evaluation_stage.py" --run-root "$RUNROOT" --phase canary
fi

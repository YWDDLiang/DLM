#!/usr/bin/env bash
#SBATCH --job-name=r03-official-hull
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:10:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
export PYTHONPATH="$SOURCE/src" CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${R03_STAGE:?}" in
 canary_hull) phase=canary ;;
 pilot_hull) phase=pilot256 ;;
 *) exit 2 ;;
esac
"$PY" "$SOURCE/operations/r03_c3fd_main_20260907/run_hull_stage.py" --run-root "${R03_RUN_ROOT:?}" --phase "$phase"

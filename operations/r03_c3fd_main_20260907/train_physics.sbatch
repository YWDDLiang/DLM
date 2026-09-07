#!/usr/bin/env bash
#SBATCH --job-name=r03-existing-physics
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=02:30:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
RUNROOT="${R03_RUN_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
B0="$PROJECT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
RUN="$RUNROOT/physics_$SLURM_JOB_ID"
mkdir "$RUN"
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; exit "$rc"' ERR
"$PY" "$SOURCE/src/scripts/train_r03_physics_transfer.py" \
 --b0-checkpoint "$B0" --sources-manifest "$RUNROOT/pointer_40395/assets/PHYSICS_SOURCES.json" \
 --output-dir "$RUN/data" --prepare-only >"$RUN/prepare.out" 2>"$RUN/prepare.err"
"$PY" "$SOURCE/src/scripts/train_r03_physics_transfer.py" \
 --b0-checkpoint "$B0" --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
 --prepared-dir "$RUN/data" --output-dir "$RUN/train" --updates 128 --effective-batch 16 \
 --seed 20260907 >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
touch "$RUN/_SUCCESS"

#!/bin/bash
#SBATCH --job-name=h1-r03e-assemble
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUN="$ROOT/runs/20260802_h1_body_safeaxis_refined_repeats4_v1"
: "${R03E_SOURCE_SHA256:?R03E source manifest SHA is required}"

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE/runtime:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONHASHSEED=0

for REPEAT in 0 1 2 3; do
  test -f "$RUN/repeats/$REPEAT/_SUCCESS"
done

python "$SOURCE/assemble_terminal.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --run-root "$RUN" \
  --source-manifest-sha256 "$R03E_SOURCE_SHA256"

test -f "$RUN/_SUCCESS"

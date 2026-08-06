#!/bin/bash
#SBATCH --job-name=h1a2c-sun256-gen-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-1%2
#SBATCH --output=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%A_%a.out
#SBATCH --error=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%A_%a.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1"
RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1"
: "${H1A2C_SUN256_SOURCE_SHA256:?source manifest SHA256 is required}"
arms=(P0 P1)
ARM="${arms[${SLURM_ARRAY_TASK_ID:?array task ID is required}]}"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1A2C_SUN256_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "generation identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0

mkdir -p "$RUN/arms/$ARM"
python "$SOURCE/run_paired_body_refine.py" \
  --arm "$ARM" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --data-dir "$RUN/data" \
  --output-dir "$RUN/arms/$ARM/generation" \
  --execution-manifest-sha256 "$H1A2C_SUN256_SOURCE_SHA256"

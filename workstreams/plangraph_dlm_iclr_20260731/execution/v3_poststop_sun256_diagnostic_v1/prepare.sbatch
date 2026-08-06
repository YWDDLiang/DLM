#!/bin/bash
#SBATCH --job-name=v3-sun256-prep
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_diagnostic_v1"
RUN="$ROOT/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1"
: "${V3_SUN_SOURCE_SHA256:?execution source manifest SHA is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != short ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$V3_SUN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "preparation identity/resource contract changed" >&2
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
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0

test ! -e "$RUN/data"
python "$SOURCE/prepare_plans.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --output-dir "$RUN/data" \
  --execution-manifest-sha256 "$V3_SUN_SOURCE_SHA256"


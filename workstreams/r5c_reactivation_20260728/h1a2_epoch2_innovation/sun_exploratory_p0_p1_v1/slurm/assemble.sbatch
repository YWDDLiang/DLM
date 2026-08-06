#!/bin/bash
#SBATCH --job-name=h1a2c-sun256-final-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%j.out
#SBATCH --error=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1"
RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1"
: "${H1A2C_SUN256_SOURCE_SHA256:?source manifest SHA256 is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != short ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ ! "$H1A2C_SUN256_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "assembly identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONHASHSEED=0

python "$SOURCE/assemble_terminal.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --data-dir "$RUN/data" \
  --arms-root "$RUN/arms" \
  --output-dir "$RUN/final" \
  --execution-manifest-sha256 "$H1A2C_SUN256_SOURCE_SHA256"

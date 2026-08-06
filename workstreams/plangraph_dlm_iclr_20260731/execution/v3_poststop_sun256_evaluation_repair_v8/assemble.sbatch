#!/bin/bash
#SBATCH --job-name=v3-sun256-v8-asm
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8/logs/%x-%j.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_evaluation_repair_v8"
RUN="$ROOT/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8"
: "${V8_SUN_SOURCE_SHA256:?v8 source manifest SHA is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != short ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ ! "$V8_SUN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "assembly identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE"
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
  --input-manifest "$SOURCE/input_manifest.json" \
  --arms-root "$RUN/arms" \
  --output "$RUN/terminal_report.json" \
  --source-manifest-sha256 "$V8_SUN_SOURCE_SHA256"

#!/bin/bash
#SBATCH --job-name=h1-plan-ledger-preflight-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_real_ledger_plan_preflight_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_real_ledger_plan_preflight_v1/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_real_ledger_plan_preflight_v1"
RUN="$ROOT/runs/20260802_h1_real_ledger_plan_preflight_v1"
: "${H1_PLAN_PREFLIGHT_SOURCE_SHA256:?source manifest SHA256 is required}"

if [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ -n "${SLURM_JOB_GPUS:-}" ]] \
  || [[ "${SLURM_GPUS_ON_NODE:-0}" != "0" ]] \
  || [[ ! "$H1_PLAN_PREFLIGHT_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "CPU preflight resource or source identity changed" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" != "$H1_PLAN_PREFLIGHT_SOURCE_SHA256" ]]; then
  echo "preflight source manifest identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python "$SOURCE/real_ledger_preflight.py" \
  --config "$SOURCE/CONFIG.json" \
  --output-dir "$RUN/preflight"

#!/bin/bash
#SBATCH --job-name=h1-dlm-b3-panels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1/logs/%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1/logs/%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
PANEL_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v1/panels"
PANEL_EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_state_panels_v1"
B3_CHECKPOINT="${RUN_ROOT}/arms/B3/output/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

record_failure() {
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '%s\n' "${rc}" > "${RUN_ROOT}/status/score_${SLURM_JOB_ID}_FAILED"
  fi
  exit "${rc}"
}
trap record_failure EXIT

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "registered B3 scoring Slurm identity changed" >&2
  exit 2
fi

test -d "${SOURCE_ROOT}"
test -d "${B3_CHECKPOINT}"
test -f "${RUN_ROOT}/status/training_terminal_report.json"
test -f "${RUN_ROOT}/status/B3_adapter.sha256"
test -f "${RUN_ROOT}/status/frozen_panel_manifest.sha256"
test -f "${PANEL_ROOT}/terminal_report.json"
test ! -e "${RUN_ROOT}/state_panel_score"
cd "${SOURCE_ROOT}"
sha256sum -c "${RUN_ROOT}/status/source_files.sha256"
sha256sum -c "${RUN_ROOT}/status/frozen_panel_manifest.sha256"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/score_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

PANEL_SHA="$(cut -d' ' -f1 "${RUN_ROOT}/status/frozen_panel_manifest.sha256")"
B3_SHA="$(cut -d' ' -f1 "${RUN_ROOT}/status/B3_adapter.sha256")"

"${PYTHON}" "${PANEL_EXECUTION_DIR}/score_frozen_state_panels.py" \
  --config "${PANEL_EXECUTION_DIR}/CONFIG.json" \
  --arm B3 \
  --checkpoint "${B3_CHECKPOINT}" \
  --checkpoint-adapter-sha256 "${B3_SHA}" \
  --frozen-panel-root "${PANEL_ROOT}" \
  --frozen-panel-manifest-sha256 "${PANEL_SHA}" \
  --output-dir "${RUN_ROOT}/state_panel_score"

touch "${RUN_ROOT}/status/score_${SLURM_JOB_ID}_SUCCESS"
trap - EXIT
echo "b3_state_panel_score_complete=${SLURM_JOB_ID}"
echo "automatic_body64_submission=false"

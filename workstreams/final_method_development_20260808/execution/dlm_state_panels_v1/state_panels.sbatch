#!/bin/bash
#SBATCH --job-name=h1-dlm-panels-b0-v5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_state_panels_b0_v5/logs/%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_state_panels_b0_v5/logs/%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v5"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_state_panels_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

record_exit() {
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '%s\n' "${rc}" > "${RUN_ROOT}/status/job_${SLURM_JOB_ID}_FAILED"
  fi
  exit "${rc}"
}
trap record_exit EXIT

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "registered state-panel Slurm identity changed" >&2
  exit 2
fi

test -d "${SOURCE_ROOT}"
test -f "${RUN_ROOT}/status/source_files.sha256"
test -d "${RUN_ROOT}/status/submission.lock"
test ! -e "${RUN_ROOT}/panels"
cd "${SOURCE_ROOT}"
sha256sum -c "${RUN_ROOT}/status/source_files.sha256"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

"${PYTHON}" - <<'PY'
import torch
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

"${PYTHON}" "${EXECUTION_DIR}/evaluate_state_panels.py" \
  --config "${EXECUTION_DIR}/CONFIG.json" \
  --output-dir "${RUN_ROOT}/panels"

"${PYTHON}" - "${RUN_ROOT}/panels/terminal_report.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = report.get("summary", {})
producer_audit = report.get("producer_rescore_audit", {})
if (
    report.get("status") != "complete"
    or set(summary) != {"iid", "d1", "safe_axis_synthetic", "safe_axis_actual_b0"}
    or summary["iid"].get("states") != 100
    or summary["d1"].get("states") != 600
    or summary["safe_axis_synthetic"].get("validation_rows") != 100
    or summary["safe_axis_actual_b0"].get("validation_rows") != 64
    or producer_audit.get("status") != "pass"
    or producer_audit.get("batch_geometry")
    != "exact_original_rollout_batch_members_and_order"
    or producer_audit.get("max_abs_delta_threshold") != 0.0005
    or report.get("scientific_score_batch_size") != 1
    or report.get("automatic_b3_submission") is not False
):
    raise SystemExit("state-panel terminal gate failed")
PY

touch "${RUN_ROOT}/status/job_${SLURM_JOB_ID}_SUCCESS"
trap - EXIT
echo "state_panels_complete=${SLURM_JOB_ID}"
echo "automatic_b3_submission=false"

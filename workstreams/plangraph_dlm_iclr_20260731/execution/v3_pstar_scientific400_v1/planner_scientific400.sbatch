#!/bin/bash
#SBATCH --job-name=v3-pstar-sci400
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_pstar_scientific400_v1"
AUTHORIZATION="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json"
MANIFEST="${EXECUTION_DIR}/EXECUTION_MANIFEST.json"
DATA_DIR="${PROJECT_ROOT}/runs/20260801_h1a2_pstar_data_preflight_v1/data/full"
P0_CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
BASE_MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

EXPECTED_AUTH_SHA256=4220233e1b968e7a8c464554d7d1fa48c8ae7febe6d9fab62ff4433a55626fbd
EXPECTED_MANIFEST_SHA256=410367a5905ede08fd1e9406bb75eab0d7cce3fd32ed8eae5259e88604f6ef55
EXPECTED_DATA_MANIFEST_SHA256=4c2ba66923ccea16244cbc7ec138e69e07f50c4bca6ec7bfa1c631cd7f406c4d

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 1 ]; then
  echo "registered Planner scientific Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=pcontrol ;;
  1) ARM=pstar ;;
  *) exit 2 ;;
esac

ARM_ROOT="${RUN_ROOT}/arms/${ARM}"
GPU_CSV="${RUN_ROOT}/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"

test -d "${SOURCE_ROOT}"
test -d "${DATA_DIR}"
test -d "${P0_CHECKPOINT}"
test -d "${BASE_MODEL}"
test ! -e "${ARM_ROOT}"
test "$(sha256sum "${AUTHORIZATION}" | cut -d' ' -f1)" = "${EXPECTED_AUTH_SHA256}"
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
test "$(sha256sum "${DATA_DIR}/manifest.json" | cut -d' ' -f1)" = "${EXPECTED_DATA_MANIFEST_SHA256}"

cd "${SOURCE_ROOT}"
sha256sum -c "${EXECUTION_DIR}/SOURCE_SHA256.txt"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN

"${PYTHON}" - <<'PY'
import os
import torch

names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "999"))
if len(names) != 1 or "A800" not in names[0] or cpus != 8:
    raise SystemExit(f"Planner resource identity changed: {names}, cpus={cpus}")
print({"gpu": names, "cpus": cpus, "torch": torch.__version__})
PY

mkdir -p "${RUN_ROOT}/arms"
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 2 > "${GPU_CSV}" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

"${PYTHON}" scripts/llama_h1a2_lookahead_sft.py \
  --arm "${ARM}" \
  --model-path "${BASE_MODEL}" \
  --checkpoint-path "${P0_CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${ARM_ROOT}" \
  --max-length 768 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-updates 400 \
  --lr 2e-6 \
  --weight-decay 0 \
  --warmup-steps 25 \
  --eval-steps 50 \
  --logging-steps 10 \
  --seed 17

cleanup
trap - EXIT

"${PYTHON}" - "${ARM_ROOT}/training_report.json" "${ARM}" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "complete"
    or report.get("arm") != sys.argv[2]
    or report.get("microbatches") != 3200
    or report.get("optimizer_updates") != 400
    or len(report.get("checkpoints") or []) != 8
):
    raise SystemExit("Planner scientific arm report gate failed")
print(f"planner_scientific_arm_gate_passed={sys.argv[2]}")
PY

echo "planner_scientific_arm_complete=${ARM}"
echo "planner_scientific_automatic_downstream=false"

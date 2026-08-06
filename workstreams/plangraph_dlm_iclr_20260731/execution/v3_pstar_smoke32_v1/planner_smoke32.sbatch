#!/bin/bash
#SBATCH --job-name=v3-pstar-smoke32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_smoke32_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_smoke32_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_pstar_smoke32_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
DATA_RUN="${PROJECT_ROOT}/runs/20260801_h1a2_pstar_data_preflight_v1"
DATA_DIR="${DATA_RUN}/data/smoke32"
P0_CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
BASE_MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_MIN:-99}" -ne 0 ] \
  || [ "${SLURM_ARRAY_TASK_MAX:-99}" -ne 1 ] \
  || [ "${SLURM_ARRAY_TASK_COUNT:-99}" -ne 2 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 1 ]; then
  echo "registered P* smoke Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=pcontrol ;;
  1) ARM=pstar ;;
  *) exit 2 ;;
esac

test -d "${SOURCE_ROOT}"
test -d "${DATA_DIR}"
test -d "${P0_CHECKPOINT}"
test -d "${BASE_MODEL}"
test -f "${DATA_RUN}/terminal_report.json"
test -f "${DATA_RUN}/preflight/smoke32/_SUCCESS"
test ! -e "${RUN_ROOT}/arms/${ARM}"

cd "${SOURCE_ROOT}"
sha256sum -c \
  workstreams/plangraph_dlm_iclr_20260731/execution/v3_pstar_smoke32_v1/SOURCE_SHA256.txt

"${PYTHON}" - "${DATA_RUN}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
terminal = json.loads((root / "terminal_report.json").read_text(encoding="utf-8"))
success = json.loads(
    (root / "preflight/smoke32/_SUCCESS").read_text(encoding="utf-8")
)
if (
    terminal.get("status") != "complete"
    or terminal.get("automatic_downstream") is not False
    or terminal.get("scientific_training_authorized") is not False
    or success.get("status") != "complete"
):
    raise SystemExit("CPU data/tokenizer preflight gate did not pass")
print("v3_pstar_smoke_data_gate=PASS")
PY

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

names = [
    torch.cuda.get_device_name(index)
    for index in range(torch.cuda.device_count())
]
cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "999"))
if len(names) != 1 or "A800" not in names[0] or cpus > 8:
    raise SystemExit(f"registered A800/CPU smoke identity changed: {names}, {cpus}")
print({"gpu": names, "cpus": cpus, "torch": torch.__version__})
PY

mkdir -p "${RUN_ROOT}/arms"
GPU_CSV="${RUN_ROOT}/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"
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
  --output-dir "${RUN_ROOT}/arms/${ARM}" \
  --max-length 768 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-updates 4 \
  --lr 2e-6 \
  --weight-decay 0 \
  --warmup-steps 1 \
  --eval-steps 4 \
  --logging-steps 1 \
  --seed 17 \
  --engineering-smoke

cleanup
trap - EXIT

"${PYTHON}" scripts/validate_h1a2_pstar_smoke.py \
  --output-dir "${RUN_ROOT}/arms/${ARM}" \
  --arm "${ARM}" \
  --gpu-csv "${GPU_CSV}"

echo "v3_pstar_smoke_arm_complete=${ARM}"
echo "v3_pstar_smoke_automatic_downstream=false"

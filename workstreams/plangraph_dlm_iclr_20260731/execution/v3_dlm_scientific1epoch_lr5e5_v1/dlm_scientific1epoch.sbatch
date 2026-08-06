#!/bin/bash
#SBATCH --job-name=v3-dlm-sci1e
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dlm_scientific1epoch_lr5e5_v1"
AUTHORIZATION="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json"
PROTOCOL_OVERRIDE="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json"
MANIFEST="${EXECUTION_DIR}/EXECUTION_MANIFEST.json"
DATA_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_sidecar_preflight_v1/data/h1a2_r5c_plangraph_sidecar_v2"
BASE_MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
R5C_CHECKPOINT="${PROJECT_ROOT}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

EXPECTED_AUTH_SHA256=4220233e1b968e7a8c464554d7d1fa48c8ae7febe6d9fab62ff4433a55626fbd
EXPECTED_PROTOCOL_SHA256=248386b9499ed8aaffa5587d28b7ba9bfea9b03f61eb9cc12f2e3311bb6017ed
EXPECTED_MANIFEST_SHA256=f4a465ec9e83c1ec886bec8959269e66eb922dcfc3856bd66cc8f983c659bc0e
EXPECTED_DATA_MANIFEST_SHA256=b38f6e414893b086fa03e5d049a4d3105a4ac153e6baf68faabe6b2a1ff4616b

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 1 ]; then
  echo "registered DLM scientific Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=B1; POLICY=d1 ;;
  1) ARM=B2; POLICY=d2 ;;
  *) exit 2 ;;
esac

ARM_ROOT="${RUN_ROOT}/arms/${ARM}"
OUTPUT_DIR="${ARM_ROOT}/output"
GPU_CSV="${RUN_ROOT}/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"

test -d "${SOURCE_ROOT}"
test -d "${DATA_ROOT}"
test -d "${BASE_MODEL}"
test -d "${R5C_CHECKPOINT}"
test ! -e "${ARM_ROOT}"
test "$(sha256sum "${AUTHORIZATION}" | cut -d' ' -f1)" = "${EXPECTED_AUTH_SHA256}"
test "$(sha256sum "${PROTOCOL_OVERRIDE}" | cut -d' ' -f1)" = "${EXPECTED_PROTOCOL_SHA256}"
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
test "$(sha256sum "${DATA_ROOT}/manifest.json" | cut -d' ' -f1)" = "${EXPECTED_DATA_MANIFEST_SHA256}"
test "$(stat -c %s "${R5C_CHECKPOINT}/adapter_model.safetensors")" = "6391016776"
test "$(sha256sum "${R5C_CHECKPOINT}/adapter_config.json" | cut -d' ' -f1)" = "2b69a62cc62fdb89ecc8178a6a6fdd6bc03cac702d7bcef2dd282ea7a0bca868"
test "$(sha256sum "${R5C_CHECKPOINT}/tokenizer.json" | cut -d' ' -f1)" = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"

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
import torch
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if len(names) != 2 or any("A800" not in name for name in names):
    raise SystemExit(f"expected exactly two A800 devices, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

mkdir -p "${ARM_ROOT}"
nvidia-smi \
  --id="${CUDA_VISIBLE_DEVICES}" \
  --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 2 > "${GPU_CSV}" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

MASTER_PORT=$((23000 + SLURM_ARRAY_JOB_ID % 20000 + SLURM_ARRAY_TASK_ID))
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  scripts/llada_sft.py \
  --model-path "${BASE_MODEL}" \
  --checkpoint-path "${R5C_CHECKPOINT}" \
  --data-dir "${DATA_ROOT}" \
  --representation dynamic_v1 \
  --planned-corruption-policy "${POLICY}" \
  --iid-fraction 2 \
  --planned-fraction 1 \
  --corruption-seed 20260731 \
  --data-seed 20260515 \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 1696 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-steps 100 \
  --min-lr-ratio 0.2 \
  --weight-decay 0 \
  --logging-steps 25 \
  --eval-before-train \
  --eval-steps 212 \
  --eval-max-batches 50 \
  --save-steps 999999 \
  --dataloader-num-workers 0

cleanup
trap - EXIT

"${PYTHON}" - "${OUTPUT_DIR}" "${ARM}" "${POLICY}" <<'PY'
import json
from pathlib import Path
import sys

root, arm, policy = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
runtime = json.loads((root / "distributed_runtime_report.json").read_text(encoding="utf-8"))
if (
    config.get("planned_corruption_policy") != policy
    or config.get("lr") != 5e-05
    or config.get("max_train_steps") != 1696
    or runtime.get("status") != "complete"
    or runtime.get("optimizer_updates") != 1696
    or runtime.get("global_train_sequences") != 27136
    or runtime.get("final_checkpoint_saved") is not True
    or not (root / "final").is_dir()
):
    raise SystemExit(f"DLM scientific arm report gate failed: {arm}")
print(f"dlm_scientific_arm_gate_passed={arm}")
PY

echo "dlm_scientific_arm_complete=${ARM}"
echo "dlm_scientific_automatic_downstream=false"

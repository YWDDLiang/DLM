#!/bin/bash
#SBATCH --job-name=h1-dlm-b3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1/logs/%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1/logs/%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_b3_v1"
DATA_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_sidecar_preflight_v1/data/h1a2_r5c_plangraph_sidecar_v2"
BASE_MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
B0_CHECKPOINT="${PROJECT_ROOT}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
OUTPUT_DIR="${RUN_ROOT}/arms/B3/output"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
GPU_CSV="${RUN_ROOT}/logs/${SLURM_JOB_ID}.gpu.csv"

record_failure() {
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '%s\n' "${rc}" > "${RUN_ROOT}/status/train_${SLURM_JOB_ID}_FAILED"
  fi
  exit "${rc}"
}
trap record_failure EXIT

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "registered B3 Slurm identity changed" >&2
  exit 2
fi

test -d "${SOURCE_ROOT}"
test -d "${DATA_ROOT}"
test -d "${BASE_MODEL}"
test -d "${B0_CHECKPOINT}"
test -d "${RUN_ROOT}/status/submission.lock"
test ! -e "${RUN_ROOT}/arms/B3"
test "$(sha256sum "${DATA_ROOT}/manifest.json" | cut -d' ' -f1)" = "b38f6e414893b086fa03e5d049a4d3105a4ac153e6baf68faabe6b2a1ff4616b"
test "$(stat -c %s "${B0_CHECKPOINT}/adapter_model.safetensors")" = "6391016776"
test "$(sha256sum "${B0_CHECKPOINT}/adapter_config.json" | cut -d' ' -f1)" = "2b69a62cc62fdb89ecc8178a6a6fdd6bc03cac702d7bcef2dd282ea7a0bca868"
test "$(sha256sum "${B0_CHECKPOINT}/tokenizer.json" | cut -d' ' -f1)" = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"
cd "${SOURCE_ROOT}"
sha256sum -c "${RUN_ROOT}/status/source_files.sha256"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/train_${SLURM_JOB_ID}"
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

mkdir -p "${RUN_ROOT}/arms/B3"
nvidia-smi \
  --id="${CUDA_VISIBLE_DEVICES}" \
  --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 2 > "${GPU_CSV}" &
MONITOR_PID=$!
cleanup_monitor() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
record_training_exit() {
  rc=$?
  cleanup_monitor
  if [ "${rc}" -ne 0 ]; then
    printf '%s\n' "${rc}" > "${RUN_ROOT}/status/train_${SLURM_JOB_ID}_FAILED"
  fi
  exit "${rc}"
}
trap record_training_exit EXIT

MASTER_PORT=$((23000 + SLURM_JOB_ID % 20000))
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  scripts/llada_sft.py \
  --model-path "${BASE_MODEL}" \
  --checkpoint-path "${B0_CHECKPOINT}" \
  --data-dir "${DATA_ROOT}" \
  --representation dynamic_v1 \
  --planned-corruption-policy d2_safe_axis \
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

cleanup_monitor
trap record_failure EXIT

ADAPTER_SHA="$(sha256sum "${OUTPUT_DIR}/final/adapter_model.safetensors" | cut -d' ' -f1)"
printf '%s  %s\n' "${ADAPTER_SHA}" "${OUTPUT_DIR}/final/adapter_model.safetensors" \
  > "${RUN_ROOT}/status/B3_adapter.sha256"

"${PYTHON}" - \
  "${OUTPUT_DIR}" \
  "${RUN_ROOT}/status/B3_adapter.sha256" \
  "${RUN_ROOT}/status/training_terminal_report.json" <<'PY'
import json
from pathlib import Path
import os
import sys

root = Path(sys.argv[1])
adapter_record = Path(sys.argv[2])
output = Path(sys.argv[3])
config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
runtime = json.loads(
    (root / "distributed_runtime_report.json").read_text(encoding="utf-8")
)
adapter_sha = adapter_record.read_text(encoding="utf-8").split()[0]
gate = bool(
    config.get("representation") == "dynamic_v1"
    and config.get("planned_corruption_policy") == "d2_safe_axis"
    and config.get("iid_fraction") == 2.0
    and config.get("planned_fraction") == 1.0
    and config.get("corruption_seed") == 20260731
    and config.get("data_seed") == 20260515
    and config.get("lr") == 5e-05
    and config.get("max_train_steps") == 1696
    and runtime.get("status") == "complete"
    and runtime.get("world_size") == 2
    and runtime.get("optimizer_updates") == 1696
    and runtime.get("global_train_sequences") == 27136
    and runtime.get("final_checkpoint_saved") is True
    and (root / "final").is_dir()
)
report = {
    "schema": "evidence_first_dlm_b3_training_terminal_v1",
    "status": "complete" if gate else "failed",
    "arm": "B3",
    "initialization": "B0",
    "planned_corruption_policy": config.get("planned_corruption_policy"),
    "iid_fraction": config.get("iid_fraction"),
    "planned_fraction": config.get("planned_fraction"),
    "optimizer_updates": runtime.get("optimizer_updates"),
    "global_train_sequences": runtime.get("global_train_sequences"),
    "final_checkpoint": str(root / "final"),
    "final_adapter_sha256": adapter_sha,
    "runtime_gate_passed": runtime.get("runtime_gate_passed"),
    "automatic_body64_submission": False,
    "automatic_ratio_sweep": False,
    "automatic_downstream": False,
    "automatic_sun": False,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not gate:
    raise SystemExit("B3 fixed-endpoint training gate failed")
print(json.dumps(report, indent=2, sort_keys=True))
PY

touch "${RUN_ROOT}/status/train_${SLURM_JOB_ID}_SUCCESS"
trap - EXIT
echo "b3_training_complete=${SLURM_JOB_ID}"
echo "automatic_body64_submission=false"

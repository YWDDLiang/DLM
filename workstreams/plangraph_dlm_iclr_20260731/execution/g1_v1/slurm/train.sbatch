#!/usr/bin/env bash
#SBATCH --job-name=pg-g1-train-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_g1_v1/slurm/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_g1_v1/slurm/%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260731_plangraph_dlm_g1_v1"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/g1_source_v1"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
MANIFEST="${EXECUTION_ROOT}/G1_EXECUTION_MANIFEST_V1.json"
AUTHORIZATION="${EXECUTION_ROOT}/G1_AUTHORIZATION_V1.json"
DATASET_ROOT="${PROJECT_ROOT}/data/dlm_sft/mp_20_plangraph_v1_20260731"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
INITIAL_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
INITIAL_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) ARM=PG; DATA_DIR="${DATASET_ROOT}/planner" ;;
  1) ARM=PG-shuffle; DATA_DIR="${RUN_ROOT}/data/PG-shuffle" ;;
  *) echo "unexpected G1 training arm" >&2; exit 2 ;;
esac
OUTPUT_DIR="${RUN_ROOT}/training/${ARM}"

trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/train_${ARM}_exit_status.txt"; nvidia-smi > "${RUN_ROOT}/status/train_${ARM}_gpu_end.txt" 2>&1 || true; exit "${status}"' EXIT
test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test "${SLURM_ARRAY_TASK_COUNT:-0}" -eq 2
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
test -d "${MODEL_PATH}"
test -d "${INITIAL_ADAPTER}"
test -s "${RUN_ROOT}/data/ledger_report.json"
test -d "${DATA_DIR}"
test ! -e "${OUTPUT_DIR}"

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/train_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN

"${PYTHON}" -c 'import os,torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; assert len(names)==1 and "A800" in names[0] and int(os.environ.get("SLURM_CPUS_PER_TASK","99"))<=8, names; print({"gpu":names})'
nvidia-smi > "${RUN_ROOT}/status/train_${ARM}_gpu_start.txt"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader,nounits -l 2 > "${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv" &
MONITOR_PID=$!
cleanup_monitor() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap 'status=$?; cleanup_monitor; printf "%s\n" "${status}" > "${RUN_ROOT}/status/train_'"${ARM}"'_exit_status.txt"; nvidia-smi > "${RUN_ROOT}/status/train_'"${ARM}"'_gpu_end.txt" 2>&1 || true; exit "${status}"' EXIT

"${PYTHON}" "${SOURCE_ROOT}/train_plangraph_planner.py" \
  --arm "${ARM}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${INITIAL_ADAPTER}" \
  --initial-adapter-sha256 "${INITIAL_ADAPTER_SHA256}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 1024 \
  --max-updates 400 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-6 \
  --warmup-updates 25 \
  --weight-decay 0 \
  --eval-steps 50 \
  --eval-rows 64 \
  --logging-steps 10 \
  --seed 17

test -s "${OUTPUT_DIR}/training_report.json"
echo "G1_TRAIN_ACCEPTANCE=PASS ARM=${ARM}"


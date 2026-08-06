#!/bin/bash
#SBATCH --job-name=v3-dlm-2g-smoke32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dlm_b1_b2_2xa800_smoke32_v1"
AUTHORIZATION="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/AUTHORIZATION_V3_DLM_2XA800_SMOKE32_20260801.json"
MANIFEST="${EXECUTION_DIR}/EXECUTION_MANIFEST.json"
DATA_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_sidecar_preflight_v1/data/h1a2_r5c_plangraph_sidecar_v2"
BASE_MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/
R5C_CHECKPOINT="${PROJECT_ROOT}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

EXPECTED_AUTH_SHA256=7786f43727ffdcd8cbddcaf9818f73f616de82f779d7d3f45497f1a58e4a8a29
EXPECTED_MANIFEST_SHA256=10c6d802e437a2ccf4b6fa0c54aa52f48a75f404fe2d24749cbfff5082f1c8b5
EXPECTED_DATA_MANIFEST_SHA256=b38f6e414893b086fa03e5d049a4d3105a4ac153e6baf68faabe6b2a1ff4616b
EXPECTED_DATA_SUCCESS_SHA256=c29f04bd45b033520d35232d9f3a7653f1c4ec6cb1d0f8ce0c9b7daf08e6ce42

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 1 ]; then
  echo "two-A800 smoke Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=B1; POLICY=d1 ;;
  1) ARM=B2; POLICY=d2 ;;
  *) echo "unknown array task" >&2; exit 2 ;;
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
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
test "$(sha256sum "${DATA_ROOT}/manifest.json" | cut -d' ' -f1)" = "${EXPECTED_DATA_MANIFEST_SHA256}"
test "$(sha256sum "${DATA_ROOT}/_SUCCESS" | cut -d' ' -f1)" = "${EXPECTED_DATA_SUCCESS_SHA256}"
test "$(head -n 32 "${DATA_ROOT}/train.jsonl" | sha256sum | cut -d' ' -f1)" = "0c190b8182b6b7dafa3798ced95ebb744f163feb268464003a85b6e04d0882fb"
test "$(head -n 32 "${DATA_ROOT}/val.jsonl" | sha256sum | cut -d' ' -f1)" = "bc7c7aa210be1e837e128811dc31f6cd4dc74363f6654287ab63005b1550e293"
test "$(stat -c %s "${R5C_CHECKPOINT}/adapter_model.safetensors")" = "6391016776"
test "$(sha256sum "${R5C_CHECKPOINT}/adapter_config.json" | cut -d' ' -f1)" = "2b69a62cc62fdb89ecc8178a6a6fdd6bc03cac702d7bcef2dd282ea7a0bca868"
test "$(sha256sum "${R5C_CHECKPOINT}/tokenizer_config.json" | cut -d' ' -f1)" = "8e89acaa54a8fb8fc7d228165ac483f61b7fef7c4c9761214092511190f75de2"
test "$(sha256sum "${R5C_CHECKPOINT}/special_tokens_map.json" | cut -d' ' -f1)" = "95c306b5886d915f970629a0fe34e30e20fb8805a31f694846d01ea4487e8790"
test "$(sha256sum "${R5C_CHECKPOINT}/tokenizer.json" | cut -d' ' -f1)" = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"

mkdir -p "${ARM_ROOT}"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

cd "${SOURCE_ROOT}"
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

"${PYTHON}" -m unittest \
  tests.test_distributed_data \
  tests.test_llada_distributed_runtime

"${PYTHON}" - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible CUDA devices, observed {torch.cuda.device_count()}")
for index in range(2):
    if "A800" not in torch.cuda.get_device_name(index):
        raise SystemExit(f"visible CUDA device {index} is not A800: {torch.cuda.get_device_name(index)}")
print("two_a800_visibility_gate_passed=true")
PY

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
  --max-train-steps 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --lr-scheduler constant \
  --warmup-steps 0 \
  --weight-decay 0 \
  --logging-steps 1 \
  --eval-before-train \
  --eval-steps 2 \
  --eval-max-batches 16 \
  --save-steps 999999 \
  --dataloader-num-workers 0 \
  --limit-train 32 \
  --limit-val 32 \
  --engineering-only \
  --skip-final-save

cleanup
trap - EXIT

"${PYTHON}" "${EXECUTION_DIR}/assemble_arm_report.py" \
  --arm "${ARM}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpu-csv "${GPU_CSV}" \
  --output "${ARM_ROOT}/engineering_report.json"

echo "h1a2_dlm_2xa800_smoke_arm_complete=${ARM}"
echo "h1a2_dlm_2xa800_smoke_automatic_downstream=false"

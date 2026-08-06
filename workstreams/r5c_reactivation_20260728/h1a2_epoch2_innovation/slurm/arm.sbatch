#!/usr/bin/env bash
#SBATCH --job-name=h1a2c-jc-arm-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2%2
#SBATCH --output=runs/20260729_h1a2c_jointchem_v1/logs/%x-%A_%a.out
#SBATCH --error=runs/20260729_h1a2c_jointchem_v1/logs/%x-%A_%a.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
WORK_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation"
RESTORED_BASELINE_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/baseline"
if [ -d "${RESTORED_BASELINE_ROOT}/crystal_dlm" ]; then
  RUNTIME_ROOT="${RESTORED_BASELINE_ROOT}"
else
  RUNTIME_ROOT="${PROJECT_ROOT}"
fi
RUN_ROOT="${PROJECT_ROOT}/runs/20260729_h1a2c_jointchem_v1"
PYTHON="${PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
EPOCH2_ADAPTER="${EPOCH2_ADAPTER:-${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final}"
SOURCE_DIR="${PROJECT_ROOT}/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base"
DATA_DIR="${RUN_ROOT}/data"
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
PATCH_SHA256="${H1A2C_JOINTCHEM_PATCH_SHA256:?H1A2C_JOINTCHEM_PATCH_SHA256 is required}"
EPOCH2_ADAPTER_SHA256="65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"

case "${TASK_ID}" in
  0) ARM="P0"; TRAIN_ARM="" ;;
  1) ARM="P1"; TRAIN_ARM="valid_replay" ;;
  2) ARM="P2"; TRAIN_ARM="jointchem" ;;
  *) echo "unexpected task ${TASK_ID}" >&2; exit 2 ;;
esac

ARM_ROOT="${RUN_ROOT}/arms/${ARM}"
cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/arm_${ARM}_exit_status.txt"; nvidia-smi > "${RUN_ROOT}/status/arm_${ARM}_gpu_end.txt" 2>&1 || true; exit "${status}"' EXIT

test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test -x "${PYTHON}"
test -d "${MODEL_PATH}"
test -d "${EPOCH2_ADAPTER}"
test -s "${DATA_DIR}/_SUCCESS"
test ! -e "${ARM_ROOT}"
(cd "${WORK_ROOT}" && test "$(sha256sum SOURCE_SHA256.txt | awk '{print $1}')" = "${PATCH_SHA256}")
(cd "${WORK_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
(cd "${RUNTIME_ROOT}" && sha256sum -c "${WORK_ROOT}/RUNTIME_REQUIRED_SHA256.txt")
test "$(sha256sum "${EPOCH2_ADAPTER}/adapter_model.safetensors" | awk '{print $1}')" = "${EPOCH2_ADAPTER_SHA256}"

export PYTHONPATH="${WORK_ROOT}/code:${RUNTIME_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" -c 'import os, torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; cpus=int(os.environ.get("SLURM_CPUS_PER_TASK","999")); assert len(names)==1 and "A800" in names[0] and cpus<=8, (names,cpus); print({"gpu":names,"cpu":cpus})'
nvidia-smi > "${RUN_ROOT}/status/arm_${ARM}_gpu_start.txt"
"${PYTHON}" "${WORK_ROOT}/code/preflight_jointchem.py" \
  --tokenizer-source "${EPOCH2_ADAPTER}" \
  --output "${RUN_ROOT}/status/arm_${ARM}_loss_preflight.json" \
  --require-one-a800
mkdir -p "${ARM_ROOT}"

CHECKPOINT="${EPOCH2_ADAPTER}"
CHECKPOINT_IDENTITY_SHA256="${EPOCH2_ADAPTER_SHA256}"
STEP=0
if [ -n "${TRAIN_ARM}" ]; then
  "${PYTHON}" "${WORK_ROOT}/code/train_h1a2_jointchem.py" \
    --arm "${TRAIN_ARM}" \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${EPOCH2_ADAPTER}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ARM_ROOT}/training" \
    --max-length 768 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-updates 400 \
    --lr 2e-6 \
    --weight-decay 0 \
    --warmup-steps 25 \
    --logging-steps 10 \
    --eval-steps 50 \
    --eval-max-batches 128 \
    --seed 17 \
    --margin 0.10 \
    --sft-weight 0.75 \
    --chemistry-weight 0.15 \
    --joint-weight 0.10 \
    --execution-manifest-sha256 "${PATCH_SHA256}" \
    --initial-adapter-sha256 "${EPOCH2_ADAPTER_SHA256}" \
    --require-one-a800

  "${PYTHON}" "${WORK_ROOT}/code/select_jointchem.py" checkpoint \
    --training-report "${ARM_ROOT}/training/training_report.json" \
    --output "${ARM_ROOT}/checkpoint_selection.json"
  CHECKPOINT="$("${PYTHON}" -c 'import json,sys; value=json.load(open(sys.argv[1])); selected=value.get("selected"); print("" if not isinstance(selected,dict) else selected["checkpoint_dir"])' "${ARM_ROOT}/checkpoint_selection.json")"
  CHECKPOINT_IDENTITY_SHA256="$("${PYTHON}" -c 'import json,sys; value=json.load(open(sys.argv[1])); selected=value.get("selected"); print("" if not isinstance(selected,dict) else selected["checkpoint_manifest_sha256"])' "${ARM_ROOT}/checkpoint_selection.json")"
  STEP="$("${PYTHON}" -c 'import json,sys; value=json.load(open(sys.argv[1])); selected=value.get("selected"); print(-1 if not isinstance(selected,dict) else selected["step"])' "${ARM_ROOT}/checkpoint_selection.json")"
  if [ -z "${CHECKPOINT}" ]; then
    echo "ARM_ACCEPTANCE=SCIENTIFIC_STOP_NO_ELIGIBLE_CHECKPOINT"
    exit 0
  fi
fi

"${PYTHON}" "${RUNTIME_ROOT}/scripts/sample_llama_h1_formula_plans.py" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${ARM_ROOT}/plan512" \
  --num-samples 512 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --seed 17029

"${PYTHON}" "${WORK_ROOT}/code/evaluate_jointchem_plans.py" \
  --identity "${ARM}" \
  --arm "${TRAIN_ARM:-frozen_epoch2}" \
  --step "${STEP}" \
  --raw-generations "${ARM_ROOT}/plan512/raw_generations.jsonl" \
  --teacher-jsonl "${SOURCE_DIR}/train.jsonl" \
  --sample-metrics "${ARM_ROOT}/plan512/sample_metrics.json" \
  --denominator 512 \
  --execution-manifest-sha256 "${PATCH_SHA256}" \
  --checkpoint-identity-sha256 "${CHECKPOINT_IDENTITY_SHA256}" \
  --initial-adapter-sha256 "${EPOCH2_ADAPTER_SHA256}" \
  --output "${ARM_ROOT}/plan_report.json"

echo "ARM_ACCEPTANCE=PASS ARM=${ARM} STEP=${STEP}"

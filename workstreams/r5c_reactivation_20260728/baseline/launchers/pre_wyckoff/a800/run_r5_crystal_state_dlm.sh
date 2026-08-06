#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
STAGE="${STAGE:-r5b}"

TEXT_DATA_DIR="${TEXT_DATA_DIR:-data/dlm_sft/mp_20_crysllmgen_text}"
MODULAR_DATA_DIR="${MODULAR_DATA_DIR:-data/dlm_sft/mp_20_crysllmgen_modular_v2}"
EXACT_DATA_DIR="${EXACT_DATA_DIR:-data/dlm_sft/mp_20_r5_exact_length}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2 for this project." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((21000 + (${SLURM_JOB_ID:-0} % 30000)))}"
PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((BASE_MASTER_PORT + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "===== COMMAND $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  printf '%q ' "$@" | tee -a "${log_file}"
  echo | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== STATUS ${status} $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  return "${status}"
}

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "${STAGE}",
  "project_root": "${PROJECT_ROOT}",
  "model_path": "${MODEL_PATH}",
  "gpu_count": int("${GPU_COUNT}"),
  "constraints": {
    "max_a800_gpus": 2,
    "a800_execution": "slurm_only",
    "source_of_truth": "local_machine",
    "transfer_policy": "sftp_or_scp_only_no_rsync"
  }
}
Path("${NOTES_DIR}/r5_crystal_state_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/r5_repair.py \
    crystal_dlm/r5_verifier.py \
    scripts/r5_data_audit.py \
    scripts/build_r5_plan_state_sft_data.py \
    scripts/build_r5_exact_length_sft_data.py \
    scripts/build_r5_repair_sft_data.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/rank_r5_proposals.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest tests.test_crysllmgen_text tests.test_r5_conditioning tests.test_r5_dynamic_length tests.test_r5_plan_state tests.test_r5_repair_verifier

if [ "${STAGE}" = "r5_0" ] || [ "${STAGE}" = "all" ]; then
  run_logged "${LOG_DIR}/r5_0_data_audit.log" \
    python scripts/r5_data_audit.py \
      --output-dir "${NOTES_DIR}"
fi

if [ "${STAGE}" = "r5b" ] || [ "${STAGE}" = "all" ]; then
  if [ ! -f "${MODULAR_DATA_DIR}/_SUCCESS" ]; then
    run_logged "${LOG_DIR}/build_crysllmgen_modular_data.log" \
      python scripts/build_crysllmgen_modular_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${MODULAR_DATA_DIR}" \
        --tokenizer-path "${MODEL_PATH}" \
        --skip-graph-validation \
        --module-style composition \
        --site-coord-rows sampled-one
  fi
  MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${MODULAR_DATA_DIR}/stats.json").read_text())
print(min(768, max(256, int(stats.get("max_length_recommended") or 384) + 32)))
PY
)
  next_port
  run_logged "${LOG_DIR}/r5b_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --data-dir "${MODULAR_DATA_DIR}" \
      --representation crysllmgen_text \
      --skip-data-vocab-resize \
      --output-dir "${OUT_DIR}/r5b_modular_sft" \
      --max-length "${MAX_LENGTH}" \
      --epochs 1 \
      --batch-size "${TRAIN_BATCH_SIZE}" \
      --grad-accum "${TRAIN_GRAD_ACCUM}" \
      --lr "${TRAIN_LR}" \
      --lr-scheduler cosine \
      --warmup-steps "${TRAIN_WARMUP_STEPS}" \
      --min-lr-ratio 0.2 \
      --logging-steps 20 \
      --eval-steps 500 \
      --eval-max-batches 50 \
      --save-steps 999999 \
      --modules-to-save "" \
      --save-embedding-layers false \
      --crysllmgen-composition-loss-weight 2.5
  next_port
  run_logged "${LOG_DIR}/r5b_sample256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_crysllmgen_modular.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${OUT_DIR}/r5b_modular_sft/final" \
      --output-dir "${OUT_DIR}/r5b_sample256" \
      --num-samples "${SMOKE_SAMPLES}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --temperature "${TEMPERATURE}" \
      --module-style composition
  run_logged "${LOG_DIR}/r5b_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-generations-jsonl "${OUT_DIR}/r5b_sample256/raw_generations.jsonl" \
      --representation crysllmgen_text \
      --output-json "${NOTES_DIR}/r5b_sample256_composition.json" \
      --output-md "${NOTES_DIR}/r5b_sample256_composition.md"
fi

if [ "${STAGE}" = "r5c" ] || [ "${STAGE}" = "all" ]; then
  if [ ! -f "${EXACT_DATA_DIR}/_SUCCESS" ]; then
    run_logged "${LOG_DIR}/build_r5_exact_length_data.log" \
      python scripts/build_r5_exact_length_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${EXACT_DATA_DIR}" \
        --tokenizer-path "${MODEL_PATH}"
  fi
  MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${EXACT_DATA_DIR}/stats.json").read_text())
print(min(768, max(256, int(stats.get("max_length_recommended") or 384) + 32)))
PY
)
  next_port
  run_logged "${LOG_DIR}/r5c_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --data-dir "${EXACT_DATA_DIR}" \
      --representation dynamic_v1 \
      --output-dir "${OUT_DIR}/r5c_exact_sft" \
      --max-length "${MAX_LENGTH}" \
      --epochs 1 \
      --batch-size "${TRAIN_BATCH_SIZE}" \
      --grad-accum "${TRAIN_GRAD_ACCUM}" \
      --lr "${TRAIN_LR}" \
      --lr-scheduler cosine \
      --warmup-steps "${TRAIN_WARMUP_STEPS}" \
      --min-lr-ratio 0.2 \
      --logging-steps 20 \
      --eval-steps 500 \
      --eval-max-batches 50 \
      --save-steps 999999
  next_port
  run_logged "${LOG_DIR}/r5c_sample256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${OUT_DIR}/r5c_exact_sft/final" \
      --prompt-jsonl "${EXACT_DATA_DIR}/test.jsonl" \
      --output-dir "${OUT_DIR}/r5c_sample256" \
      --num-samples "${SMOKE_SAMPLES}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --temperature "${TEMPERATURE}"
fi

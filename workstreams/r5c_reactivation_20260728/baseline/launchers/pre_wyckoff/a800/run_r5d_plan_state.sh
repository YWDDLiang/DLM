#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_state}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
REBUILD_PLAN_DATA="${REBUILD_PLAN_DATA:-0}"

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
  "stage": "r5d_plan_state",
  "project_root": "${PROJECT_ROOT}",
  "model_path": "${MODEL_PATH}",
  "plan_data_dir": "${PLAN_DATA_DIR}",
  "smoke_samples": int("${SMOKE_SAMPLES}"),
  "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
  "temperature": float("${TEMPERATURE}"),
  "weighted_sampling": True,
  "constraints": {
    "max_a800_gpus": 2,
    "a800_execution": "slurm_only",
    "source_of_truth": "local_machine",
    "transfer_policy": "sftp_or_scp_only_no_rsync"
  }
}
Path("${NOTES_DIR}/r5d_plan_state_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/composition_validity.py \
    scripts/build_r5_plan_state_sft_data.py \
    scripts/sample_llada_r5_plan_state.py \
    scripts/evaluate_r5d_plan_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest tests.test_r5_plan_state

if [ "${REBUILD_PLAN_DATA}" = "1" ] || [ ! -f "${PLAN_DATA_DIR}/_SUCCESS" ]; then
  if [ "${REBUILD_PLAN_DATA}" = "1" ]; then
    rm -rf "${PLAN_DATA_DIR}"
  fi
  run_logged "${LOG_DIR}/build_r5_plan_state_data.log" \
    python scripts/build_r5_plan_state_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${PLAN_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --progress-every 2000
fi

MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${PLAN_DATA_DIR}/stats.json").read_text())
print(min(768, max(256, int(stats.get("max_length_recommended") or 384) + 32)))
PY
)
GEN_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${PLAN_DATA_DIR}/stats.json").read_text())
print(max(64, int(stats.get("answer_token_count") or stats.get("max_answer_model_length") or 192) + 16))
PY
)

next_port
run_logged "${LOG_DIR}/r5d_plan_train.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${PLAN_DATA_DIR}" \
    --representation r5_plan_state \
    --skip-data-vocab-resize \
    --output-dir "${OUT_DIR}/r5d_plan_sft" \
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
    --weighted-sampling \
    --sample-weight-multipliers strict=1.4,all_metal=0.55,single_element=0.05,invalid=0.35

next_port
run_logged "${LOG_DIR}/r5d_plan_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_plan_state.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/r5d_plan_sft/final" \
    --stats-json "${PLAN_DATA_DIR}/stats.json" \
    --output-dir "${OUT_DIR}/r5d_plan_sample256" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --gen-length "${GEN_LENGTH}"

run_logged "${LOG_DIR}/r5d_plan_gate.log" \
  python scripts/evaluate_r5d_plan_gate.py \
    --sample-metrics "${OUT_DIR}/r5d_plan_sample256/sample_metrics.json" \
    --raw-generations-jsonl "${OUT_DIR}/r5d_plan_sample256/raw_generations.jsonl" \
    --output-json "${NOTES_DIR}/r5d_plan_sample256_gate.json" \
    --output-md "${NOTES_DIR}/r5d_plan_sample256_gate_report.md"

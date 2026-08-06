#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260529_234235-r5d-planstate-256}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/${PARENT_RUN_ID}/outputs/r5d_plan_sft/final}"
PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_state}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
GEN_LENGTH="${GEN_LENGTH:-}"
STEPS="${STEPS:-}"
BLOCK_LENGTH="${BLOCK_LENGTH:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((21000 + (${SLURM_JOB_ID:-0} % 30000)))}"

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

if [ -z "${GEN_LENGTH}" ]; then
  GEN_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${PLAN_DATA_DIR}/stats.json").read_text())
print(max(64, int(stats.get("answer_token_count") or stats.get("max_answer_model_length") or 212)))
PY
)
fi

if [ -z "${STEPS}" ]; then
  STEPS="${GEN_LENGTH}"
fi

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "r5d2_plan_state_resample",
  "parent_run_id": "${PARENT_RUN_ID}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "model_path": "${MODEL_PATH}",
  "plan_data_dir": "${PLAN_DATA_DIR}",
  "smoke_samples": int("${SMOKE_SAMPLES}"),
  "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
  "temperature": float("${TEMPERATURE}"),
  "gen_length": int("${GEN_LENGTH}"),
  "steps": int("${STEPS}"),
  "block_length": int("${BLOCK_LENGTH}"),
  "hypothesis": "low-temperature shorter-generation resample tests whether free JSON plan failure is sampling noise or over-generation"
}
Path("${NOTES_DIR}/r5d2_plan_resample_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/composition_validity.py \
    scripts/sample_llada_r5_plan_state.py \
    scripts/evaluate_r5d_plan_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest tests.test_r5_plan_state

run_logged "${LOG_DIR}/r5d2_plan_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${BASE_MASTER_PORT}" scripts/sample_llada_r5_plan_state.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --stats-json "${PLAN_DATA_DIR}/stats.json" \
    --output-dir "${OUT_DIR}/r5d2_plan_sample256" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --gen-length "${GEN_LENGTH}" \
    --steps "${STEPS}" \
    --block-length "${BLOCK_LENGTH}"

run_logged "${LOG_DIR}/r5d2_plan_gate.log" \
  python scripts/evaluate_r5d_plan_gate.py \
    --sample-metrics "${OUT_DIR}/r5d2_plan_sample256/sample_metrics.json" \
    --raw-generations-jsonl "${OUT_DIR}/r5d2_plan_sample256/raw_generations.jsonl" \
    --output-json "${NOTES_DIR}/r5d2_plan_sample256_gate.json" \
    --output-md "${NOTES_DIR}/r5d2_plan_sample256_gate_report.md"

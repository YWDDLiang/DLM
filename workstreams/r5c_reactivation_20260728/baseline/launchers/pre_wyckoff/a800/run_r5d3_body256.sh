#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PLAN_RUN_ID="${PLAN_RUN_ID:-20260530_012145-r5d3-plancompact-256}"
PLAN_JSONL="${PLAN_JSONL:-runs/${PLAN_RUN_ID}/outputs/r5d3_plan_sample256/raw_generations.jsonl}"
BODY_PARENT_RUN_ID="${BODY_PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/${BODY_PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
SAMPLE_DIR="${OUT_DIR}/r5d3_body_sample256"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${SAMPLE_DIR}"

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

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "r5d3_plan_to_exact_body_256",
  "plan_run_id": "${PLAN_RUN_ID}",
  "plan_jsonl": "${PLAN_JSONL}",
  "body_parent_run_id": "${BODY_PARENT_RUN_ID}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "model_path": "${MODEL_PATH}",
  "smoke_samples": int("${SMOKE_SAMPLES}"),
  "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
  "temperature": float("${TEMPERATURE}"),
  "sampler_constraints": {
    "prefill_count_token": True,
    "freeze_plan_composition": True,
    "duplicate_coordinate_mask": True,
    "lattice_volume_mask": True
  }
}
Path("${NOTES_DIR}/r5d3_body256_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/lattice_geometry.py \
    crystal_dlm/llada_generation.py \
    crystal_dlm/cif_lite.py \
    crystal_dlm/crysllmgen_text.py \
    crystal_dlm/fixed_plain.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5c_gate.py

run_logged "${LOG_DIR}/r5d3_body_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${BASE_MASTER_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --prompt-jsonl "${PLAN_JSONL}" \
    --output-dir "${SAMPLE_DIR}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

composition_args=(
  python scripts/analyze_composition_validity.py
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl"
  --representation dynamic_v1
  --output-json "${NOTES_DIR}/r5d3_body_sample256_composition.json"
  --output-md "${NOTES_DIR}/r5d3_body_sample256_composition.md"
)
if [ -f "${SAMPLE_DIR}/raw_dlm_samples.pt" ]; then
  composition_args+=(--raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt")
fi
run_logged "${LOG_DIR}/r5d3_body_composition.log" "${composition_args[@]}"

run_logged "${LOG_DIR}/r5d3_body_gate.log" \
  python scripts/evaluate_r5c_gate.py \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --composition-summary "${NOTES_DIR}/r5d3_body_sample256_composition.json" \
    --composition-key raw_jsonl \
    --output-json "${NOTES_DIR}/r5d3_body_sample256_gate.json" \
    --output-md "${NOTES_DIR}/r5d3_body_sample256_gate_report.md"

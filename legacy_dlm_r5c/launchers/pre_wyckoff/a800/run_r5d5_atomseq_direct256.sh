#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-20260530_0906-r5e2-normrepair-direct256}"
START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-runs/${START_PARENT_RUN_ID}/outputs/r5e_plan_compact_trainrepair_sft/final}"
PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_atomseq_smact_v1}"
PLAN_FORMAT="${PLAN_FORMAT:-atomseq}"
PLAN_NAME="${PLAN_NAME:-r5d5_${PLAN_FORMAT}}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
BODY_PARENT_RUN_ID="${BODY_PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
BODY_CHECKPOINT_PATH="${BODY_CHECKPOINT_PATH:-runs/${BODY_PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
COMPOSITION_FILTER="${COMPOSITION_FILTER:-all}"
SAMPLE_WEIGHT_PROFILE="${SAMPLE_WEIGHT_PROFILE:-uniform}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-80}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
BODY_BATCH_SIZE="${BODY_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.55}"
REBUILD_PLAN_DATA="${REBUILD_PLAN_DATA:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
PLAN_SAMPLE_DIR="${OUT_DIR}/${PLAN_NAME}_plan_sample256_direct"
BODY_SAMPLE_DIR="${OUT_DIR}/${PLAN_NAME}_body_sample256"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${PLAN_SAMPLE_DIR}" "${BODY_SAMPLE_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((23000 + (${SLURM_JOB_ID:-0} % 30000)))}"
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

assert_gate_passed() {
  local gate_json="$1"
  local label="$2"
  python - "$gate_json" "$label" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("passed"):
    print(f"{sys.argv[2]} gate failed: {payload.get('failures')}", file=sys.stderr)
    raise SystemExit(1)
print(f"{sys.argv[2]} gate passed")
PY
}

python - <<PY
import json
from pathlib import Path
checkpoint_path = "${START_CHECKPOINT_PATH}"
payload = {
    "run_id": "${RUN_ID}",
    "stage": "${PLAN_NAME}_plan_direct256_body256",
    "start_checkpoint_path": None if checkpoint_path in ("", "none", "NONE", "null", "NULL") else checkpoint_path,
    "plan_data_dir": "${PLAN_DATA_DIR}",
    "plan_format": "${PLAN_FORMAT}",
    "composition_filter": "${COMPOSITION_FILTER}",
    "sample_weight_profile": "${SAMPLE_WEIGHT_PROFILE}",
    "train_epochs": int("${TRAIN_EPOCHS}"),
    "body_checkpoint_path": "${BODY_CHECKPOINT_PATH}",
    "smoke_samples": int("${SMOKE_SAMPLES}"),
    "sampling_policy": "direct_one_generation_per_sample; plan representation derives N/counts; full MP-20 training distribution by default; no candidate pool; no K/N/composition condition prior; no weighted sampling",
}
Path("${NOTES_DIR}/${PLAN_NAME}_direct256_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/lattice_geometry.py \
    scripts/build_r5_plan_atomseq_sft_data.py \
    scripts/sample_llada_r5_plan_compact.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5d_plan_gate.py \
    scripts/evaluate_r5c_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest tests.test_r5_plan_state tests.test_r5_conditioning tests.test_r5_repair_verifier

if [ "${REBUILD_PLAN_DATA}" = "1" ] || [ ! -f "${PLAN_DATA_DIR}/_SUCCESS" ]; then
  if [ "${REBUILD_PLAN_DATA}" = "1" ]; then
    rm -rf "${PLAN_DATA_DIR}"
  fi
  run_logged "${LOG_DIR}/build_${PLAN_NAME}_data.log" \
    python scripts/build_r5_plan_atomseq_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${PLAN_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --plan-format "${PLAN_FORMAT}" \
      --composition-filter "${COMPOSITION_FILTER}" \
      --sample-weight-profile "${SAMPLE_WEIGHT_PROFILE}" \
      --progress-every 2000
fi

MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${PLAN_DATA_DIR}/stats.json").read_text())
print(min(768, max(192, int(stats.get("max_length_recommended") or 256) + 24)))
PY
)

next_port
train_cmd=(
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py
  --model-path "${MODEL_PATH}"
)
if [ -n "${START_CHECKPOINT_PATH}" ] && [ "${START_CHECKPOINT_PATH}" != "none" ] && [ "${START_CHECKPOINT_PATH}" != "NONE" ] && [ "${START_CHECKPOINT_PATH}" != "null" ] && [ "${START_CHECKPOINT_PATH}" != "NULL" ]; then
  train_cmd+=(--checkpoint-path "${START_CHECKPOINT_PATH}")
fi
train_cmd+=(
  --data-dir "${PLAN_DATA_DIR}"
  --representation r5_plan_state
  --skip-data-vocab-resize
  --output-dir "${OUT_DIR}/${PLAN_NAME}_plan_sft"
  --max-length "${MAX_LENGTH}"
  --epochs "${TRAIN_EPOCHS}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --grad-accum "${TRAIN_GRAD_ACCUM}"
  --lr "${TRAIN_LR}"
  --lr-scheduler cosine
  --warmup-steps "${TRAIN_WARMUP_STEPS}"
  --min-lr-ratio 0.2
  --logging-steps 20
  --eval-steps 500
  --eval-max-batches 50
  --save-steps 999999
  --modules-to-save ""
  --save-embedding-layers false
)
run_logged "${LOG_DIR}/${PLAN_NAME}_plan_train.log" "${train_cmd[@]}"

next_port
run_logged "${LOG_DIR}/${PLAN_NAME}_plan_sample256_direct.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_plan_compact.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/${PLAN_NAME}_plan_sft/final" \
    --stats-json "${PLAN_DATA_DIR}/stats.json" \
    --output-dir "${PLAN_SAMPLE_DIR}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --plan-format "${PLAN_FORMAT}" \
    --direct-samples

run_logged "${LOG_DIR}/${PLAN_NAME}_plan_gate256_direct.log" \
  python scripts/evaluate_r5d_plan_gate.py \
    --sample-metrics "${PLAN_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${PLAN_SAMPLE_DIR}/raw_generations.jsonl" \
    --min-decoded-samples "${SMOKE_SAMPLES}" \
    --min-parse-rate 0.95 \
    --min-valid-n 0.99 \
    --min-valid-formula 0.99 \
    --min-valid-plan 0.95 \
    --min-smact-plausible 0.90 \
    --min-unique-formula 128 \
    --min-unique-prototype 96 \
    --output-json "${NOTES_DIR}/${PLAN_NAME}_plan_sample256_direct_gate.json" \
    --output-md "${NOTES_DIR}/${PLAN_NAME}_plan_sample256_direct_gate_report.md"
assert_gate_passed "${NOTES_DIR}/${PLAN_NAME}_plan_sample256_direct_gate.json" "${PLAN_NAME} plan256"

next_port
run_logged "${LOG_DIR}/${PLAN_NAME}_body_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BODY_CHECKPOINT_PATH}" \
    --prompt-jsonl "${PLAN_SAMPLE_DIR}/raw_generations.jsonl" \
    --output-dir "${BODY_SAMPLE_DIR}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${BODY_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

composition_args=(
  python scripts/analyze_composition_validity.py
  --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl"
  --representation dynamic_v1
  --output-json "${NOTES_DIR}/${PLAN_NAME}_body_sample256_composition.json"
  --output-md "${NOTES_DIR}/${PLAN_NAME}_body_sample256_composition.md"
)
if [ -f "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt" ]; then
  composition_args+=(--raw-pt "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt")
fi
run_logged "${LOG_DIR}/${PLAN_NAME}_body_composition256.log" "${composition_args[@]}"

run_logged "${LOG_DIR}/${PLAN_NAME}_body_gate256.log" \
  python scripts/evaluate_r5c_gate.py \
    --sample-metrics "${BODY_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl" \
    --composition-summary "${NOTES_DIR}/${PLAN_NAME}_body_sample256_composition.json" \
    --composition-key raw_jsonl \
    --output-json "${NOTES_DIR}/${PLAN_NAME}_body_sample256_gate.json" \
    --output-md "${NOTES_DIR}/${PLAN_NAME}_body_sample256_gate_report.md"
assert_gate_passed "${NOTES_DIR}/${PLAN_NAME}_body_sample256_gate.json" "${PLAN_NAME} body256"

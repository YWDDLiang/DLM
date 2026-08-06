#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260530_2043-r5d11-countfields-all-direct256}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-runs/${SOURCE_RUN_ID}/outputs/r5d11_countfields_all_plan_sft/final}"
ROLLOUT_RAW_JSONL="${ROLLOUT_RAW_JSONL:-runs/${SOURCE_RUN_ID}/outputs/r5d11_countfields_all_plan_sample256_direct/raw_generations.jsonl}"
PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_countfields_all_v2}"
PLAN_FORMAT="${PLAN_FORMAT:-countfields}"
PLAN_NAME="${PLAN_NAME:-r5d12_countfields_tracerl}"
REWARD_MODE="${REWARD_MODE:-reason_v1}"
BODY_PARENT_RUN_ID="${BODY_PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
BODY_CHECKPOINT_PATH="${BODY_CHECKPOINT_PATH:-runs/${BODY_PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-100}"
TRACE_LR="${TRACE_LR:-5e-7}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-2}"
TRACE_GRAD_ACCUM="${TRACE_GRAD_ACCUM:-8}"
TRACE_SHRINK="${TRACE_SHRINK:-4}"
TRACE_MAX_STATES="${TRACE_MAX_STATES:-24}"
TRACE_CLIP_EPS="${TRACE_CLIP_EPS:-0.2}"
TRACE_BETA="${TRACE_BETA:-0.02}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
BODY_BATCH_SIZE="${BODY_BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.55}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
ROLLOUT_DIR="${OUT_DIR}/rollout"
TRACE_DIR="${OUT_DIR}/${PLAN_NAME}_trace_rl"
PLAN_SAMPLE_DIR="${OUT_DIR}/${PLAN_NAME}_plan_sample256_direct"
BODY_SAMPLE_DIR="${OUT_DIR}/${PLAN_NAME}_body_sample256"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${ROLLOUT_DIR}" "${TRACE_DIR}" "${PLAN_SAMPLE_DIR}" "${BODY_SAMPLE_DIR}"

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
payload = {
    "run_id": "${RUN_ID}",
    "stage": "${PLAN_NAME}_all_rollout_tracerl_direct256_body256_if_plan_passes",
    "source_run_id": "${SOURCE_RUN_ID}",
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "rollout_raw_jsonl": "${ROLLOUT_RAW_JSONL}",
    "plan_data_dir": "${PLAN_DATA_DIR}",
    "plan_format": "${PLAN_FORMAT}",
    "reward_mode": "${REWARD_MODE}",
    "trace_max_steps": int("${TRACE_MAX_STEPS}"),
    "trace_lr": float("${TRACE_LR}"),
    "body_checkpoint_path": "${BODY_CHECKPOINT_PATH}",
    "sampling_policy": "TraceRL over every D11 direct rollout with reason rewards; no candidate pool, no rejection/retry, no weighted sampling, no training filter, no K/N/composition condition prior, no verifier selection",
}
Path("${NOTES_DIR}/${PLAN_NAME}_direct256_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/reward_r5_plan_rollouts.py \
    scripts/llada_trace_rl.py \
    scripts/sample_llada_r5_plan_compact.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_r5d_plan_gate.py \
    scripts/evaluate_r5c_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest tests.test_r5_plan_reward tests.test_rl_utils tests.test_r5_plan_state tests.test_r5_conditioning

run_logged "${LOG_DIR}/${PLAN_NAME}_reward_rollout.log" \
  python scripts/reward_r5_plan_rollouts.py \
    --input-jsonl "${ROLLOUT_RAW_JSONL}" \
    --output-jsonl "${ROLLOUT_DIR}/r5_plan_rollout_scored.jsonl" \
    --summary-json "${NOTES_DIR}/${PLAN_NAME}_reward_summary.json" \
    --summary-md "${NOTES_DIR}/${PLAN_NAME}_reward_summary.md" \
    --reward-mode "${REWARD_MODE}"

next_port
run_logged "${LOG_DIR}/${PLAN_NAME}_trace_rl.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_trace_rl.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --rollout-jsonl "${ROLLOUT_DIR}/r5_plan_rollout_scored.jsonl" \
    --data-dir "${PLAN_DATA_DIR}" \
    --output-dir "${TRACE_DIR}" \
    --max-length 325 \
    --skip-data-vocab-resize \
    --max-train-steps "${TRACE_MAX_STEPS}" \
    --batch-size "${TRACE_BATCH_SIZE}" \
    --grad-accum "${TRACE_GRAD_ACCUM}" \
    --lr "${TRACE_LR}" \
    --lr-scheduler constant \
    --clip-eps "${TRACE_CLIP_EPS}" \
    --beta "${TRACE_BETA}" \
    --trace-shrink "${TRACE_SHRINK}" \
    --max-trace-states-per-sample "${TRACE_MAX_STATES}" \
    --logging-steps 10 \
    --save-steps 999999 \
    --modules-to-save ""

next_port
run_logged "${LOG_DIR}/${PLAN_NAME}_plan_sample256_direct.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_plan_compact.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${TRACE_DIR}/final" \
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

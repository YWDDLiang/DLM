#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-20260530_053527-r5d4-trainweighted-direct256}"
START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-runs/${START_PARENT_RUN_ID}/outputs/r5d4_plan_compact_trainweighted_sft/final}"
PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_compact_trainrepair_v1}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
BODY_PARENT_RUN_ID="${BODY_PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
BODY_CHECKPOINT_PATH="${BODY_CHECKPOINT_PATH:-runs/${BODY_PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-2e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-120}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
BODY_BATCH_SIZE="${BODY_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.65}"
REPAIR_TEMPERATURE="${REPAIR_TEMPERATURE:-0.45}"
WEIGHT_PROFILE="${WEIGHT_PROFILE:-mp20_num_elements}"
REPAIR_AUGMENTATIONS_PER_ROW="${REPAIR_AUGMENTATIONS_PER_ROW:-2}"
REPAIR_SAMPLE_WEIGHT="${REPAIR_SAMPLE_WEIGHT:-2.0}"
REBUILD_PLAN_DATA="${REBUILD_PLAN_DATA:-1}"
WEIGHTED_SAMPLING="${WEIGHTED_SAMPLING:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
PLAN_SAMPLE_DIR="${OUT_DIR}/r5e_plan_sample256_direct"
PLAN_REPAIR_DIR="${OUT_DIR}/r5e_plan_sample256_repaired"
BODY_SAMPLE_DIR="${OUT_DIR}/r5e_body_sample256"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${PLAN_SAMPLE_DIR}" "${PLAN_REPAIR_DIR}" "${BODY_SAMPLE_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 30000)))}"
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

gate_passed_flag() {
  local gate_json="$1"
  python - "$gate_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if payload.get("passed") else "0")
PY
}

python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "stage": "r5e_train_repair_direct_de_novo_plan256_body256",
    "start_checkpoint_path": "${START_CHECKPOINT_PATH}",
    "plan_data_dir": "${PLAN_DATA_DIR}",
    "weight_profile": "${WEIGHT_PROFILE}",
    "repair_augmentations_per_row": int("${REPAIR_AUGMENTATIONS_PER_ROW}"),
    "repair_sample_weight": float("${REPAIR_SAMPLE_WEIGHT}"),
    "weighted_sampling": "${WEIGHTED_SAMPLING}" == "1",
    "body_checkpoint_path": "${BODY_CHECKPOINT_PATH}",
    "smoke_samples": int("${SMOKE_SAMPLES}"),
    "sampling_policy": "direct_one_generation_per_sample; optional one learned repair pass for flagged samples; no candidate pool; no K/N condition prior",
}
Path("${NOTES_DIR}/r5e_trainrepair_direct256_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/lattice_geometry.py \
    scripts/build_r5_plan_compact_sft_data.py \
    scripts/sample_llada_r5_plan_compact.py \
    scripts/repair_r5_plan_compact_samples.py \
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
  run_logged "${LOG_DIR}/build_r5e_trainrepair_data.log" \
    python scripts/build_r5_plan_compact_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${PLAN_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --weight-profile "${WEIGHT_PROFILE}" \
      --repair-augmentations-per-row "${REPAIR_AUGMENTATIONS_PER_ROW}" \
      --repair-sample-weight "${REPAIR_SAMPLE_WEIGHT}" \
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
plan_train_args=(
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${START_CHECKPOINT_PATH}" \
    --data-dir "${PLAN_DATA_DIR}" \
    --representation r5_plan_state \
    --skip-data-vocab-resize \
    --output-dir "${OUT_DIR}/r5e_plan_compact_trainrepair_sft" \
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
    --save-embedding-layers false
)
if [ "${WEIGHTED_SAMPLING}" = "1" ]; then
  plan_train_args+=(--weighted-sampling)
fi
run_logged "${LOG_DIR}/r5e_plan_train.log" "${plan_train_args[@]}"

next_port
run_logged "${LOG_DIR}/r5e_plan_sample256_direct.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_plan_compact.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/r5e_plan_compact_trainrepair_sft/final" \
    --stats-json "${PLAN_DATA_DIR}/stats.json" \
    --output-dir "${PLAN_SAMPLE_DIR}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --direct-samples

run_logged "${LOG_DIR}/r5e_plan_gate256_direct.log" \
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
    --output-json "${NOTES_DIR}/r5e_plan_sample256_direct_gate.json" \
    --output-md "${NOTES_DIR}/r5e_plan_sample256_direct_gate_report.md"

PLAN_FOR_BODY_DIR="${PLAN_SAMPLE_DIR}"
PLAN_GATE_FOR_BODY="${NOTES_DIR}/r5e_plan_sample256_direct_gate.json"
if [ "$(gate_passed_flag "${PLAN_GATE_FOR_BODY}")" != "1" ]; then
  next_port
  run_logged "${LOG_DIR}/r5e_plan_repair256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/repair_r5_plan_compact_samples.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${OUT_DIR}/r5e_plan_compact_trainrepair_sft/final" \
      --input-jsonl "${PLAN_SAMPLE_DIR}/raw_generations.jsonl" \
      --output-dir "${PLAN_REPAIR_DIR}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --temperature "${REPAIR_TEMPERATURE}" \
      --gen-length 128

  run_logged "${LOG_DIR}/r5e_plan_gate256_repaired.log" \
    python scripts/evaluate_r5d_plan_gate.py \
      --sample-metrics "${PLAN_REPAIR_DIR}/sample_metrics.json" \
      --raw-generations-jsonl "${PLAN_REPAIR_DIR}/raw_generations.jsonl" \
      --min-decoded-samples "${SMOKE_SAMPLES}" \
      --min-parse-rate 0.95 \
      --min-valid-n 0.99 \
      --min-valid-formula 0.99 \
      --min-valid-plan 0.95 \
      --min-smact-plausible 0.90 \
      --min-unique-formula 128 \
      --min-unique-prototype 96 \
      --output-json "${NOTES_DIR}/r5e_plan_sample256_repaired_gate.json" \
      --output-md "${NOTES_DIR}/r5e_plan_sample256_repaired_gate_report.md"
  assert_gate_passed "${NOTES_DIR}/r5e_plan_sample256_repaired_gate.json" "R5-E repaired plan256"
  PLAN_FOR_BODY_DIR="${PLAN_REPAIR_DIR}"
  PLAN_GATE_FOR_BODY="${NOTES_DIR}/r5e_plan_sample256_repaired_gate.json"
fi

python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${PLAN_GATE_FOR_BODY}").read_text(encoding="utf-8"))
metrics = payload["metrics"]
decoded = int(metrics.get("decoded_samples") or 0)
parse_rate = float(metrics.get("parse_rate") or 0.0)
valid_plan_rate = float(metrics.get("valid_plan_rate") or 0.0)
smact_rate = float(metrics.get("smact_plausible_rate") or 0.0)
if decoded != int("${SMOKE_SAMPLES}") or parse_rate < 0.95 or valid_plan_rate < 0.99 or smact_rate < 0.90:
    raise SystemExit(
        f"Body stage plan gate insufficient: decoded={decoded} parse={parse_rate:.4f} valid_plan={valid_plan_rate:.4f} smact={smact_rate:.4f}"
    )
PY

next_port
run_logged "${LOG_DIR}/r5e_body_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BODY_CHECKPOINT_PATH}" \
    --prompt-jsonl "${PLAN_FOR_BODY_DIR}/raw_generations.jsonl" \
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
  --output-json "${NOTES_DIR}/r5e_body_sample256_composition.json"
  --output-md "${NOTES_DIR}/r5e_body_sample256_composition.md"
)
if [ -f "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt" ]; then
  composition_args+=(--raw-pt "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt")
fi
run_logged "${LOG_DIR}/r5e_body_composition256.log" "${composition_args[@]}"

run_logged "${LOG_DIR}/r5e_body_gate256.log" \
  python scripts/evaluate_r5c_gate.py \
    --sample-metrics "${BODY_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl" \
    --composition-summary "${NOTES_DIR}/r5e_body_sample256_composition.json" \
    --composition-key raw_jsonl \
    --output-json "${NOTES_DIR}/r5e_body_sample256_gate.json" \
    --output-md "${NOTES_DIR}/r5e_body_sample256_gate_report.md"
assert_gate_passed "${NOTES_DIR}/r5e_body_sample256_gate.json" "R5-E body256"

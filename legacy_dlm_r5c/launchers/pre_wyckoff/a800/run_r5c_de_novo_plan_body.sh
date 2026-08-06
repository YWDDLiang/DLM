#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/${PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
STAGE="${STAGE:-dn1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-64}"
GATE_SAMPLES="${GATE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-}"
PLAN_STYLE="${PLAN_STYLE:-}"
JOINT_WEIGHT="${JOINT_WEIGHT:-1.0}"
PLAN_ONLY_WEIGHT="${PLAN_ONLY_WEIGHT:-}"
BODY_REPLAY_WEIGHT="${BODY_REPLAY_WEIGHT:-0.25}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2 for this project." >&2
  exit 2
fi

case "${STAGE}" in
  dn1)
    MIXTURE="${MIXTURE:-joint,body_replay}"
    ;;
  dn2)
    MIXTURE="${MIXTURE:-joint,plan_only,body_replay}"
    ;;
  dn3)
    MIXTURE="${MIXTURE:-joint,plan_only,body_replay}"
    PLAN_STYLE="${PLAN_STYLE:-formula_text}"
    PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-48}"
    PLAN_ONLY_WEIGHT="${PLAN_ONLY_WEIGHT:-1.5}"
    ;;
  dn4)
    MIXTURE="${MIXTURE:-joint,plan_only,body_replay}"
    PLAN_STYLE="${PLAN_STYLE:-semantic_formula_v1}"
    PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-80}"
    PLAN_ONLY_WEIGHT="${PLAN_ONLY_WEIGHT:-2.0}"
    ;;
  dn5)
    MIXTURE="${MIXTURE:-joint,plan_only,body_replay}"
    PLAN_STYLE="${PLAN_STYLE:-formula_end_v1}"
    PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-32}"
    PLAN_ONLY_WEIGHT="${PLAN_ONLY_WEIGHT:-1.0}"
    ;;
  *)
    echo "STAGE must be dn1, dn2, dn3, dn4, or dn5." >&2
    exit 2
    ;;
esac
if [ -z "${PLAN_STYLE}" ]; then
  PLAN_STYLE="formula_text"
fi
if [ -z "${PLAN_GEN_LENGTH}" ]; then
  PLAN_GEN_LENGTH="48"
fi
if [ -z "${PLAN_ONLY_WEIGHT}" ]; then
  PLAN_ONLY_WEIGHT="1.5"
fi

if [ -z "${DATA_DIR:-}" ]; then
  if [ "${PLAN_STYLE}" = "semantic_formula_v1" ]; then
    DATA_DIR="data/dlm_sft/mp_20_r5c_semantic_formula_plan_body_${STAGE}"
  elif [ "${PLAN_STYLE}" = "formula_end_v1" ]; then
    DATA_DIR="data/dlm_sft/mp_20_r5c_formula_end_plan_body_${STAGE}"
  else
    DATA_DIR="data/dlm_sft/mp_20_r5c_formula_plan_body_${STAGE}"
  fi
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
  "stage": "r5c_de_novo_composition_plan_${STAGE}",
  "model_path": "${MODEL_PATH}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "input_csv_dir": "${INPUT_CSV_DIR}",
  "data_dir": "${DATA_DIR}",
  "mixture": "${MIXTURE}",
  "sample_weights": {
    "joint": float("${JOINT_WEIGHT}"),
    "plan_only": float("${PLAN_ONLY_WEIGHT}"),
    "body_replay": float("${BODY_REPLAY_WEIGHT}")
  },
  "plan_style": "${PLAN_STYLE}",
  "smoke_samples": int("${SMOKE_SAMPLES}"),
  "gate_samples": int("${GATE_SAMPLES}"),
  "temperature": float("${TEMPERATURE}"),
  "plan_gen_length": int("${PLAN_GEN_LENGTH}"),
  "de_novo_constraints": {
    "external_plan_source": None,
    "generated_plan_executes_body": True,
    "sampling_time_filter_or_ranker": False
  }
}
Path("${NOTES_DIR}/r5c_de_novo_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_body.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/llada_generation.py \
    scripts/build_r5c_plan_body_sft_data.py \
    scripts/sample_llada_r5c_plan_body.py \
    scripts/evaluate_r5c_de_novo_gate.py \
    scripts/analyze_r5c_plan_distribution.py \
    scripts/llada_sft.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest \
    tests.test_crysllmgen_text \
    tests.test_r5_dynamic_length \
    tests.test_r5_plan_state \
    tests.test_r5_plan_body

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_r5c_de_novo_data.log" \
    python scripts/build_r5c_plan_body_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${CHECKPOINT_PATH}" \
      --mixture "${MIXTURE}" \
      --joint-weight "${JOINT_WEIGHT}" \
      --plan-only-weight "${PLAN_ONLY_WEIGHT}" \
      --body-replay-weight "${BODY_REPLAY_WEIGHT}" \
      --plan-style "${PLAN_STYLE}"
fi

MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${DATA_DIR}/stats.json").read_text())
print(min(1024, max(384, int(stats.get("max_length_recommended") or 512) + 32)))
PY
)

next_port
run_logged "${LOG_DIR}/r5c_de_novo_train.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --representation dynamic_v1 \
    --weighted-sampling \
    --output-dir "${OUT_DIR}/r5c_de_novo_sft" \
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
run_logged "${LOG_DIR}/r5c_de_novo_sample_smoke.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5c_plan_body.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/r5c_de_novo_sft/final" \
    --output-dir "${OUT_DIR}/r5c_de_novo_sample${SMOKE_SAMPLES}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --plan-gen-length "${PLAN_GEN_LENGTH}" \
    --plan-steps "${PLAN_GEN_LENGTH}" \
    --plan-style "${PLAN_STYLE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

gate_extra_args=()
if [ "${STAGE}" = "dn4" ]; then
  gate_extra_args=(--enable-distribution-gates --enable-semantic-gates)
elif [ "${STAGE}" = "dn5" ]; then
  gate_extra_args=(--enable-distribution-gates --enable-formula-end-gates)
fi

run_logged "${LOG_DIR}/r5c_de_novo_gate_smoke.log" \
  python scripts/evaluate_r5c_de_novo_gate.py \
    --sample-metrics "${OUT_DIR}/r5c_de_novo_sample${SMOKE_SAMPLES}/sample_metrics.json" \
    --raw-generations-jsonl "${OUT_DIR}/r5c_de_novo_sample${SMOKE_SAMPLES}/raw_generations.jsonl" \
    --output-json "${NOTES_DIR}/r5c_de_novo_sample${SMOKE_SAMPLES}_gate.json" \
    --output-md "${NOTES_DIR}/r5c_de_novo_sample${SMOKE_SAMPLES}_gate.md" \
    "${gate_extra_args[@]}" || true

next_port
run_logged "${LOG_DIR}/r5c_de_novo_sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5c_plan_body.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/r5c_de_novo_sft/final" \
    --output-dir "${OUT_DIR}/r5c_de_novo_sample256" \
    --num-samples "${GATE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --plan-gen-length "${PLAN_GEN_LENGTH}" \
    --plan-steps "${PLAN_GEN_LENGTH}" \
    --plan-style "${PLAN_STYLE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

run_logged "${LOG_DIR}/r5c_de_novo_gate256.log" \
  python scripts/evaluate_r5c_de_novo_gate.py \
    --sample-metrics "${OUT_DIR}/r5c_de_novo_sample256/sample_metrics.json" \
    --raw-generations-jsonl "${OUT_DIR}/r5c_de_novo_sample256/raw_generations.jsonl" \
    --output-json "${NOTES_DIR}/r5c_de_novo_sample256_gate.json" \
    --output-md "${NOTES_DIR}/r5c_de_novo_sample256_gate.md" \
    "${gate_extra_args[@]}" || true

if [ -f "${DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/r5c_de_novo_plan_distribution256.log" \
    python scripts/analyze_r5c_plan_distribution.py \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --generated-jsonl "${OUT_DIR}/r5c_de_novo_sample256/raw_generations.jsonl" \
      --output-json "${NOTES_DIR}/r5c_de_novo_plan_distribution256.json" \
      --output-md "${NOTES_DIR}/r5c_de_novo_plan_distribution256.md"
fi

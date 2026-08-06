#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_WARM_START_CHECKPOINT="${DLM_WARM_START_CHECKPOINT:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
PLANNER_CHECKPOINT_PATH="${PLANNER_CHECKPOINT_PATH:-runs/20260602_0312-h1-prefill-sft-plan/outputs/h1_llama_formula_sft/final}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
FORMULA_BODY_DATA_DIR="${FORMULA_BODY_DATA_DIR:-data/dlm_sft/mp_20_h1_formula_only_body}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
GATE_SAMPLES="${GATE_SAMPLES:-256}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
PLANNER_PROMPT_STYLE="${PLANNER_PROMPT_STYLE:-formula_prefill_v1}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-48}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.8}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DLM_SFT_BATCH_SIZE="${DLM_SFT_BATCH_SIZE:-1}"
DLM_SFT_GRAD_ACCUM="${DLM_SFT_GRAD_ACCUM:-8}"
DLM_SFT_LR="${DLM_SFT_LR:-2e-5}"
DLM_SFT_EPOCHS="${DLM_SFT_EPOCHS:-1}"
DLM_SFT_MAX_LENGTH="${DLM_SFT_MAX_LENGTH:-384}"
DLM_SFT_SAVE_STEPS="${DLM_SFT_SAVE_STEPS:-1000}"
DLM_SFT_EVAL_STEPS="${DLM_SFT_EVAL_STEPS:-200}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${PLANNER_NPROC}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ]; then
  echo "GPU_COUNT, PLANNER_NPROC, and DLM_NPROC must be <=2 for this project." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((23000 + (${SLURM_JOB_ID:-0} % 24000)))}"
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

trap 'status=$?; echo "${status}" > "${NOTES_DIR}/exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/end_time.txt"; nvidia-smi > "${NOTES_DIR}/gpu_status_end.txt" 2>&1 || true; exit "${status}"' EXIT

date "+%F %T %Z" > "${NOTES_DIR}/start_time.txt"
{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-none}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort > "${NOTES_DIR}/environment.txt"

python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "stage": "h1_formula_only_dlm_retrain_gate",
    "planner_model_path": "${PLANNER_MODEL_PATH}",
    "planner_checkpoint_path": "${PLANNER_CHECKPOINT_PATH}",
    "planner_prompt_style": "${PLANNER_PROMPT_STYLE}",
    "dlm_model_path": "${DLM_MODEL_PATH}",
    "dlm_warm_start_checkpoint": "${DLM_WARM_START_CHECKPOINT}",
    "formula_body_data_dir": "${FORMULA_BODY_DATA_DIR}",
    "gate_samples": int("${GATE_SAMPLES}"),
    "de_novo_constraints": {
        "gold_plan_at_sampling": False,
        "candidate_pool": False,
        "sampling_filter_or_topk": False,
        "planner_outputs_formula_only": True,
        "dlm_outputs_body_only": True,
        "body_prompt_style": "formula_only",
    },
}
Path("${NOTES_DIR}/h1b_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h1_llm_planner.py \
    crystal_dlm/h1_formula_only_body.py \
    crystal_dlm/r5_plan_body.py \
    crystal_dlm/r5_dynamic_length.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/build_h1_formula_only_body_sft_data.py \
    scripts/llada_sft.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/evaluate_h1_hybrid_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc \
    'python -m unittest discover -s tests -p test_r5_dynamic_length.py &&
     python -m unittest discover -s tests -p test_r5_plan_state.py &&
     python -m unittest discover -s tests -p test_r5_plan_body.py &&
     python -m unittest discover -s tests -p test_h1_llm_planner.py &&
     python -m unittest discover -s tests -p test_h1_formula_only_body.py'

if [ ! -f "${FORMULA_BODY_DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1_formula_only_body_data.log" \
    python scripts/build_h1_formula_only_body_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${FORMULA_BODY_DATA_DIR}" \
      --tokenizer-path "${DLM_MODEL_PATH}" \
      --mixture body_replay,joint_context \
      --body-replay-weight 1.0 \
      --joint-context-weight 0.25
fi

next_port
run_logged "${LOG_DIR}/h1_formula_only_dlm_sft.log" \
  torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
    --model-path "${DLM_MODEL_PATH}" \
    --checkpoint-path "${DLM_WARM_START_CHECKPOINT}" \
    --data-dir "${FORMULA_BODY_DATA_DIR}" \
    --output-dir "${OUT_DIR}/h1_formula_only_dlm_sft" \
    --representation dynamic_v1 \
    --max-length "${DLM_SFT_MAX_LENGTH}" \
    --epochs "${DLM_SFT_EPOCHS}" \
    --batch-size "${DLM_SFT_BATCH_SIZE}" \
    --grad-accum "${DLM_SFT_GRAD_ACCUM}" \
    --lr "${DLM_SFT_LR}" \
    --save-steps "${DLM_SFT_SAVE_STEPS}" \
    --eval-steps "${DLM_SFT_EVAL_STEPS}"

planner_dir="${OUT_DIR}/h1b_formula_planner256"
planner_args=()
if [ "${PLANNER_PROMPT_STYLE}" = "formula_prefill_v1" ]; then
  planner_args+=(--prompt-style formula_prefill_v1)
else
  planner_args+=(--prompt-style "${PLANNER_PROMPT_STYLE}")
fi
next_port
run_logged "${LOG_DIR}/h1b_formula_planner_sample256.log" \
  torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
    --model-path "${PLANNER_MODEL_PATH}" \
    --checkpoint-path "${PLANNER_CHECKPOINT_PATH}" \
    --output-dir "${planner_dir}" \
    --num-samples "${GATE_SAMPLES}" \
    --batch-size "${PLANNER_BATCH_SIZE}" \
    --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
    --temperature "${PLANNER_TEMPERATURE}" \
    --top-p "${PLANNER_TOP_P}" \
    --top-k "${PLANNER_TOP_K}" \
    --seed "${PLANNER_SEED}" \
    "${planner_args[@]}"

run_logged "${LOG_DIR}/h1b_planner_gate256.log" \
  python scripts/evaluate_h1_planner_gate.py \
    --sample-metrics "${planner_dir}/sample_metrics.json" \
    --raw-generations-jsonl "${planner_dir}/raw_generations.jsonl" \
    --teacher-jsonl "${FORMULA_BODY_DATA_DIR}/train.jsonl" \
    --output-json "${NOTES_DIR}/h1b_planner256_gate.json" \
    --output-md "${NOTES_DIR}/h1b_planner256_gate.md" || true

plans_jsonl="${planner_dir}/plans_for_dlm.jsonl"
plan_count=$(python - <<PY
from pathlib import Path
path = Path("${plans_jsonl}")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0)
PY
)
echo "planner_valid_plan_count=${plan_count}" | tee -a "${LOG_DIR}/h1b_formula_only_body256.log"
if [ "${plan_count}" -le 0 ]; then
  echo "No parsed planner plans available for H1-B body sampling." >&2
  exit 1
fi

body_dir="${OUT_DIR}/h1b_formula_only_body256"
next_port
run_logged "${LOG_DIR}/h1b_formula_only_body256.log" \
  torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${DLM_MODEL_PATH}" \
    --checkpoint-path "${OUT_DIR}/h1_formula_only_dlm_sft/final" \
    --prompt-jsonl "${plans_jsonl}" \
    --output-dir "${body_dir}" \
    --body-prompt-style formula_only \
    --num-samples "${plan_count}" \
    --batch-size "${DLM_BATCH_SIZE}" \
    --temperature "${DLM_TEMPERATURE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

run_logged "${LOG_DIR}/h1b_hybrid_gate256.log" \
  python scripts/evaluate_h1_hybrid_gate.py \
    --planner-gate-json "${NOTES_DIR}/h1b_planner256_gate.json" \
    --body-sample-metrics "${body_dir}/sample_metrics.json" \
    --output-json "${NOTES_DIR}/h1b_hybrid256_gate.json" \
    --output-md "${NOTES_DIR}/h1b_hybrid256_gate.md" || true

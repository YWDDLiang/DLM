#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1_llm_formula_sft}"
MODE="${MODE:-auto}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
GATE_SAMPLES="${GATE_SAMPLES:-256}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-48}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.8}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_DO_SAMPLE="${PLANNER_DO_SAMPLE:-1}"
PLANNER_STOP_AFTER_PLAN_MARKER="${PLANNER_STOP_AFTER_PLAN_MARKER:-1}"
PLANNER_TRUNCATE_AFTER_PLAN_MARKER="${PLANNER_TRUNCATE_AFTER_PLAN_MARKER:-1}"
PLANNER_SEED="${PLANNER_SEED:-17}"
PLANNER_CHECKPOINT_PATH="${PLANNER_CHECKPOINT_PATH:-}"
PLANNER_PROMPT_STYLE="${PLANNER_PROMPT_STYLE:-chat_formula_end_v1}"
PLANNER_INCLUDE_SAMPLE_ID="${PLANNER_INCLUDE_SAMPLE_ID:-1}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DLM_BODY_PROMPT_STYLE="${DLM_BODY_PROMPT_STYLE:-full_plan_state}"
RUN_SFT_IF_NEEDED="${RUN_SFT_IF_NEEDED:-1}"
RUN_HYBRID_IF_PLANNER_PASSES="${RUN_HYBRID_IF_PLANNER_PASSES:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_LR="${SFT_LR:-2e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-1.0}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-512}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2 for this project." >&2
  exit 2
fi
if [ "${PLANNER_NPROC}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ]; then
  echo "PLANNER_NPROC and DLM_NPROC must be <=2 for this project." >&2
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

BASE_MASTER_PORT="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 25000)))}"
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
    "stage": "h1_llm_plan_dlm_body_hybrid_de_novo",
    "mode": "${MODE}",
    "planner_model_path": "${PLANNER_MODEL_PATH}",
    "dlm_model_path": "${DLM_MODEL_PATH}",
    "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
    "data_dir": "${DATA_DIR}",
    "input_csv_dir": "${INPUT_CSV_DIR}",
    "gate_samples": int("${GATE_SAMPLES}"),
    "planner_max_new_tokens": int("${PLANNER_MAX_NEW_TOKENS}"),
    "planner_temperature": float("${PLANNER_TEMPERATURE}"),
    "planner_top_p": float("${PLANNER_TOP_P}"),
    "planner_top_k": int("${PLANNER_TOP_K}"),
    "planner_do_sample": "${PLANNER_DO_SAMPLE}" == "1",
    "planner_stop_after_plan_marker": "${PLANNER_STOP_AFTER_PLAN_MARKER}" == "1",
    "planner_truncate_after_plan_marker": "${PLANNER_TRUNCATE_AFTER_PLAN_MARKER}" == "1",
    "planner_seed": int("${PLANNER_SEED}"),
    "planner_checkpoint_path": "${PLANNER_CHECKPOINT_PATH}",
    "planner_prompt_style": "${PLANNER_PROMPT_STYLE}",
    "planner_include_sample_id": "${PLANNER_INCLUDE_SAMPLE_ID}" == "1",
    "dlm_temperature": float("${DLM_TEMPERATURE}"),
    "dlm_body_prompt_style": "${DLM_BODY_PROMPT_STYLE}",
    "de_novo_constraints": {
        "gold_plan_at_sampling": False,
        "candidate_pool": False,
        "sampling_filter_or_topk": False,
        "planner_outputs_formula_only": "${PLANNER_PROMPT_STYLE}" != "h1_rich_plan_v1",
        "planner_outputs_rich_plan": "${PLANNER_PROMPT_STYLE}" == "h1_rich_plan_v1",
        "dlm_outputs_body_only": True,
    },
}
Path("${NOTES_DIR}/h1_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h1_llm_planner.py \
    crystal_dlm/h1_formula_only_body.py \
    crystal_dlm/r5_plan_body.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/r5_plan_state.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/build_h1_llm_formula_sft_data.py \
    scripts/build_h1_formula_only_body_sft_data.py \
    scripts/llama_formula_sft.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_h1_hybrid_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc \
    'python -m unittest discover -s tests -p test_crysllmgen_text.py &&
     python -m unittest discover -s tests -p test_r5_dynamic_length.py &&
     python -m unittest discover -s tests -p test_r5_plan_state.py &&
     python -m unittest discover -s tests -p test_r5_plan_body.py &&
     python -m unittest discover -s tests -p test_h1_llm_planner.py &&
     python -m unittest discover -s tests -p test_h1_formula_only_body.py'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1_llm_formula_sft_data.log" \
    python scripts/build_h1_llm_formula_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${PLANNER_MODEL_PATH}" \
      --prompt-style "${PLANNER_PROMPT_STYLE}" \
      $([ "${PLANNER_INCLUDE_SAMPLE_ID}" = "1" ] && echo "--include-sample-id" || echo "--no-include-sample-id")
fi

gate_passed() {
  local gate_json="$1"
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${gate_json}").read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed_acceptable", False) else 1)
PY
}

sample_and_gate_planner() {
  local label="$1"
  local checkpoint_path="$2"
  local sample_dir="${OUT_DIR}/${label}_planner256"
  local gate_json="${NOTES_DIR}/${label}_planner256_gate.json"
  local planner_args=()
  if [ "${PLANNER_DO_SAMPLE}" != "1" ]; then
    planner_args+=(--no-do-sample)
  fi
  if [ "${PLANNER_STOP_AFTER_PLAN_MARKER}" != "1" ]; then
    planner_args+=(--no-stop-after-plan-marker)
  fi
  if [ "${PLANNER_TRUNCATE_AFTER_PLAN_MARKER}" != "1" ]; then
    planner_args+=(--no-truncate-after-plan-marker)
  fi
  if [ "${PLANNER_INCLUDE_SAMPLE_ID}" != "1" ]; then
    planner_args+=(--no-include-sample-id)
  fi
  next_port
  if [ -n "${checkpoint_path}" ]; then
    run_logged "${LOG_DIR}/${label}_planner_sample256.log" \
      torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
        --model-path "${PLANNER_MODEL_PATH}" \
        --checkpoint-path "${checkpoint_path}" \
        --output-dir "${sample_dir}" \
        --num-samples "${GATE_SAMPLES}" \
        --batch-size "${PLANNER_BATCH_SIZE}" \
        --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
        --temperature "${PLANNER_TEMPERATURE}" \
        --top-p "${PLANNER_TOP_P}" \
        --top-k "${PLANNER_TOP_K}" \
        --seed "${PLANNER_SEED}" \
        --prompt-style "${PLANNER_PROMPT_STYLE}" \
        "${planner_args[@]}"
  else
    run_logged "${LOG_DIR}/${label}_planner_sample256.log" \
      torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
        --model-path "${PLANNER_MODEL_PATH}" \
        --output-dir "${sample_dir}" \
        --num-samples "${GATE_SAMPLES}" \
        --batch-size "${PLANNER_BATCH_SIZE}" \
        --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
        --temperature "${PLANNER_TEMPERATURE}" \
        --top-p "${PLANNER_TOP_P}" \
        --top-k "${PLANNER_TOP_K}" \
        --seed "${PLANNER_SEED}" \
        --prompt-style "${PLANNER_PROMPT_STYLE}" \
        "${planner_args[@]}"
  fi

  run_logged "${LOG_DIR}/${label}_planner_gate256.log" \
    python scripts/evaluate_h1_planner_gate.py \
      --sample-metrics "${sample_dir}/sample_metrics.json" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --output-json "${gate_json}" \
      --output-md "${NOTES_DIR}/${label}_planner256_gate.md" || true
}

run_hybrid_gate() {
  local label="$1"
  local planner_gate_json="${NOTES_DIR}/${label}_planner256_gate.json"
  local plans_jsonl="${OUT_DIR}/${label}_planner256/plans_for_dlm.jsonl"
  local body_dir="${OUT_DIR}/${label}_hybrid_body256"
  local plan_count
  plan_count=$(python - <<PY
from pathlib import Path
path = Path("${plans_jsonl}")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0)
PY
)
  echo "planner_valid_plan_count=${plan_count}" | tee -a "${LOG_DIR}/${label}_hybrid_body256.log"
  if [ "${plan_count}" -le 0 ]; then
    echo "No parsed planner plans available for hybrid DLM body sampling." >&2
    return 1
  fi
  next_port
  run_logged "${LOG_DIR}/${label}_hybrid_body256.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_CHECKPOINT_PATH}" \
      --prompt-jsonl "${plans_jsonl}" \
      --output-dir "${body_dir}" \
      --body-prompt-style "${DLM_BODY_PROMPT_STYLE}" \
      --num-samples "${plan_count}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --temperature "${DLM_TEMPERATURE}" \
      --freeze-plan-composition \
      --duplicate-coordinate-mask \
      --lattice-volume-mask

  run_logged "${LOG_DIR}/${label}_hybrid_gate256.log" \
    python scripts/evaluate_h1_hybrid_gate.py \
      --planner-gate-json "${planner_gate_json}" \
      --body-sample-metrics "${body_dir}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/${label}_hybrid256_gate.json" \
      --output-md "${NOTES_DIR}/${label}_hybrid256_gate.md" || true
}

case "${MODE}" in
  zs_gate|auto)
    sample_and_gate_planner "h1_zs" ""
    if gate_passed "${NOTES_DIR}/h1_zs_planner256_gate.json"; then
      echo "H1-ZS planner passed acceptable gate." | tee -a "${LOG_DIR}/h1_decision.log"
      if [ "${RUN_HYBRID_IF_PLANNER_PASSES}" = "1" ]; then
        run_hybrid_gate "h1_zs"
      fi
    else
      echo "H1-ZS planner failed acceptable gate." | tee -a "${LOG_DIR}/h1_decision.log"
      if [ "${MODE}" = "auto" ] && [ "${RUN_SFT_IF_NEEDED}" = "1" ]; then
        run_logged "${LOG_DIR}/h1_llama_formula_sft.log" \
          python scripts/llama_formula_sft.py \
            --model-path "${PLANNER_MODEL_PATH}" \
            --data-dir "${DATA_DIR}" \
            --output-dir "${OUT_DIR}/h1_llama_formula_sft" \
            --max-length "${SFT_MAX_LENGTH}" \
            --epochs "${SFT_EPOCHS}" \
            --batch-size "${SFT_BATCH_SIZE}" \
            --grad-accum "${SFT_GRAD_ACCUM}" \
            --lr "${SFT_LR}"
        sample_and_gate_planner "h1_sft" "${OUT_DIR}/h1_llama_formula_sft/final"
        if gate_passed "${NOTES_DIR}/h1_sft_planner256_gate.json"; then
          echo "H1-SFT planner passed acceptable gate." | tee -a "${LOG_DIR}/h1_decision.log"
          if [ "${RUN_HYBRID_IF_PLANNER_PASSES}" = "1" ]; then
            run_hybrid_gate "h1_sft"
          fi
        else
          echo "H1-SFT planner failed acceptable gate; stopping before DLM body." | tee -a "${LOG_DIR}/h1_decision.log"
        fi
      fi
    fi
    ;;
  sft_gate)
    run_logged "${LOG_DIR}/h1_llama_formula_sft.log" \
      python scripts/llama_formula_sft.py \
        --model-path "${PLANNER_MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUT_DIR}/h1_llama_formula_sft" \
        --max-length "${SFT_MAX_LENGTH}" \
        --epochs "${SFT_EPOCHS}" \
        --batch-size "${SFT_BATCH_SIZE}" \
        --grad-accum "${SFT_GRAD_ACCUM}" \
        --lr "${SFT_LR}"
    sample_and_gate_planner "h1_sft" "${OUT_DIR}/h1_llama_formula_sft/final"
    if gate_passed "${NOTES_DIR}/h1_sft_planner256_gate.json" && [ "${RUN_HYBRID_IF_PLANNER_PASSES}" = "1" ]; then
      run_hybrid_gate "h1_sft"
    fi
    ;;
  planner_gate)
    LABEL="${LABEL:-h1_planner_diag}"
    if [ -z "${PLANNER_CHECKPOINT_PATH}" ]; then
      echo "MODE=planner_gate requires PLANNER_CHECKPOINT_PATH." >&2
      exit 2
    fi
    sample_and_gate_planner "${LABEL}" "${PLANNER_CHECKPOINT_PATH}"
    if gate_passed "${NOTES_DIR}/${LABEL}_planner256_gate.json" && [ "${RUN_HYBRID_IF_PLANNER_PASSES}" = "1" ]; then
      run_hybrid_gate "${LABEL}"
    fi
    ;;
  hybrid256)
    LABEL="${LABEL:-h1_sft}"
    run_hybrid_gate "${LABEL}"
    ;;
  *)
    echo "MODE must be zs_gate, sft_gate, planner_gate, hybrid256, or auto." >&2
    exit 2
    ;;
esac

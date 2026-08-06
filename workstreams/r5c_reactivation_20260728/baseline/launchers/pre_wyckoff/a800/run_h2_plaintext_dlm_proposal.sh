#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_WARM_START="${DLM_WARM_START:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
DLM_TOKENIZER_PATH="${DLM_TOKENIZER_PATH:-${DLM_WARM_START}}"
if [ -z "${DLM_TOKENIZER_PATH}" ] || [ "${DLM_TOKENIZER_PATH}" = "none" ]; then
  DLM_TOKENIZER_PATH="${DLM_MODEL_PATH}"
fi
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
PLANNER_DATA_DIR="${PLANNER_DATA_DIR:-data/dlm_sft/mp_20_h1a2_rich_planner_noid}"
H2_DATA_DIR="${H2_DATA_DIR:-data/dlm_sft/mp_20_h2_plaintext_dlm}"
PLANNER_DATA_LIMIT="${PLANNER_DATA_LIMIT:-}"
H2_DATA_LIMIT="${H2_DATA_LIMIT:-}"
H1A2_RUN_ID="${H1A2_RUN_ID:-}"
PLANNER_CHECKPOINT_PATH="${PLANNER_CHECKPOINT_PATH:-}"
if [ -z "${PLANNER_CHECKPOINT_PATH}" ] && [ -n "${H1A2_RUN_ID}" ]; then
  PLANNER_CHECKPOINT_PATH="runs/${H1A2_RUN_ID}/outputs/h1a2_llama_rich_sft/final"
fi
if [ -z "${PLANNER_CHECKPOINT_PATH}" ]; then
  echo "PLANNER_CHECKPOINT_PATH or H1A2_RUN_ID is required for H2." >&2
  exit 2
fi

GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-96}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.5}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
H2_SFT_BATCH_SIZE="${H2_SFT_BATCH_SIZE:-1}"
H2_SFT_GRAD_ACCUM="${H2_SFT_GRAD_ACCUM:-8}"
H2_SFT_LR="${H2_SFT_LR:-1e-5}"
H2_SFT_EPOCHS="${H2_SFT_EPOCHS:-1}"
H2_SFT_MAX_TRAIN_STEPS="${H2_SFT_MAX_TRAIN_STEPS:-0}"
H2_SFT_MAX_LENGTH="${H2_SFT_MAX_LENGTH:-768}"
H2_GEN_LENGTH="${H2_GEN_LENGTH:-360}"
H2_BLOCK_LENGTH="${H2_BLOCK_LENGTH:-4}"
H2_TEMPERATURE="${H2_TEMPERATURE:-0.7}"
H2_SKIP_GRAPH_VALIDATION="${H2_SKIP_GRAPH_VALIDATION:-0}"
DIFF_STEPS="${DIFF_STEPS:-800}"
RUN_DLM_IF_PLANNER_FAIL="${RUN_DLM_IF_PLANNER_FAIL:-0}"
RUN_REFINE="${RUN_REFINE:-1}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${PLANNER_NPROC}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
  echo "GPU counts must be <=2 for this project." >&2
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
REFINED_PT=""
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((25000 + (${SLURM_JOB_ID:-0} % 18000)))}"
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
  "stage": "h2_llm_plan_plaintext_dlm_proposal",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "planner_checkpoint_path": "${PLANNER_CHECKPOINT_PATH}",
  "dlm_model_path": "${DLM_MODEL_PATH}",
  "dlm_warm_start": "${DLM_WARM_START}",
  "h2_data_dir": "${H2_DATA_DIR}",
  "h2_skip_graph_validation": "${H2_SKIP_GRAPH_VALIDATION}" == "1",
  "sample_count": int("${SAMPLE_COUNT}"),
  "de_novo_constraints": {
    "gold_plan_at_sampling": False,
    "candidate_pool": False,
    "sampling_filter_or_topk": False,
    "dlm_generates_raw_structure_proposal": True,
    "dense_geometry_special_tokens": False
  }
}
Path("${NOTES_DIR}/h2_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h2_plaintext_dlm.py \
    crystal_dlm/h1_llm_planner.py \
    crystal_dlm/crysllmgen_text.py \
    scripts/build_h1_llm_formula_sft_data.py \
    scripts/build_h2_plaintext_dlm_sft_data.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/sample_llada_h2_plaintext_dlm.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/evaluate_h2_plaintext_gate.py \
    scripts/llada_sft.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h2_plaintext_dlm tests.test_h1_llm_planner tests.test_crysllmgen_text'

if [ ! -f "${PLANNER_DATA_DIR}/_SUCCESS" ]; then
  if [ -n "${PLANNER_DATA_LIMIT}" ]; then
    run_logged "${LOG_DIR}/build_h1a2_rich_planner_data_for_h2.log" \
      python scripts/build_h1_llm_formula_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${PLANNER_DATA_DIR}" \
        --tokenizer-path "${PLANNER_MODEL_PATH}" \
        --prompt-style h1_rich_plan_v1 \
        --no-include-sample-id \
        --limit "${PLANNER_DATA_LIMIT}"
  else
    run_logged "${LOG_DIR}/build_h1a2_rich_planner_data_for_h2.log" \
      python scripts/build_h1_llm_formula_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${PLANNER_DATA_DIR}" \
        --tokenizer-path "${PLANNER_MODEL_PATH}" \
        --prompt-style h1_rich_plan_v1 \
        --no-include-sample-id
  fi
fi

if [ ! -f "${H2_DATA_DIR}/_SUCCESS" ]; then
  if [ -n "${H2_DATA_LIMIT}" ] && [ "${H2_SKIP_GRAPH_VALIDATION}" = "1" ]; then
    run_logged "${LOG_DIR}/build_h2_plaintext_dlm_data.log" \
      python scripts/build_h2_plaintext_dlm_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${H2_DATA_DIR}" \
        --tokenizer-path "${DLM_TOKENIZER_PATH}" \
        --limit "${H2_DATA_LIMIT}" \
        --skip-graph-validation
  elif [ -n "${H2_DATA_LIMIT}" ]; then
    run_logged "${LOG_DIR}/build_h2_plaintext_dlm_data.log" \
      python scripts/build_h2_plaintext_dlm_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${H2_DATA_DIR}" \
        --tokenizer-path "${DLM_TOKENIZER_PATH}" \
        --limit "${H2_DATA_LIMIT}"
  elif [ "${H2_SKIP_GRAPH_VALIDATION}" = "1" ]; then
    run_logged "${LOG_DIR}/build_h2_plaintext_dlm_data.log" \
      python scripts/build_h2_plaintext_dlm_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${H2_DATA_DIR}" \
        --tokenizer-path "${DLM_TOKENIZER_PATH}" \
        --skip-graph-validation
  else
    run_logged "${LOG_DIR}/build_h2_plaintext_dlm_data.log" \
      python scripts/build_h2_plaintext_dlm_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${H2_DATA_DIR}" \
        --tokenizer-path "${DLM_TOKENIZER_PATH}"
  fi
fi

next_port
if [ -n "${DLM_WARM_START}" ] && [ "${DLM_WARM_START}" != "none" ]; then
  run_logged "${LOG_DIR}/h2_plaintext_dlm_sft.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_WARM_START}" \
      --data-dir "${H2_DATA_DIR}" \
      --output-dir "${OUT_DIR}/h2_plaintext_dlm_sft" \
      --representation crysllmgen_text \
      --max-length "${H2_SFT_MAX_LENGTH}" \
      --epochs "${H2_SFT_EPOCHS}" \
      --max-train-steps "${H2_SFT_MAX_TRAIN_STEPS}" \
      --batch-size "${H2_SFT_BATCH_SIZE}" \
      --grad-accum "${H2_SFT_GRAD_ACCUM}" \
      --lr "${H2_SFT_LR}" \
      --save-embedding-layers false
else
  run_logged "${LOG_DIR}/h2_plaintext_dlm_sft.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${DLM_MODEL_PATH}" \
      --data-dir "${H2_DATA_DIR}" \
      --output-dir "${OUT_DIR}/h2_plaintext_dlm_sft" \
      --representation crysllmgen_text \
      --max-length "${H2_SFT_MAX_LENGTH}" \
      --epochs "${H2_SFT_EPOCHS}" \
      --max-train-steps "${H2_SFT_MAX_TRAIN_STEPS}" \
      --batch-size "${H2_SFT_BATCH_SIZE}" \
      --grad-accum "${H2_SFT_GRAD_ACCUM}" \
      --lr "${H2_SFT_LR}" \
      --save-embedding-layers false
fi

PLANNER_SAMPLE_DIR="${OUT_DIR}/h2_planner256"
next_port
run_logged "${LOG_DIR}/h2_planner_sample256.log" \
  torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
    --model-path "${PLANNER_MODEL_PATH}" \
    --checkpoint-path "${PLANNER_CHECKPOINT_PATH}" \
    --output-dir "${PLANNER_SAMPLE_DIR}" \
    --num-samples "${SAMPLE_COUNT}" \
    --batch-size "${PLANNER_BATCH_SIZE}" \
    --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
    --temperature "${PLANNER_TEMPERATURE}" \
    --top-p "${PLANNER_TOP_P}" \
    --top-k "${PLANNER_TOP_K}" \
    --seed "${PLANNER_SEED}" \
    --prompt-style h1_rich_plan_v1 \
    --no-include-sample-id

run_logged "${LOG_DIR}/h2_planner_gate256.log" \
  python scripts/evaluate_h1_planner_gate.py \
    --sample-metrics "${PLANNER_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${PLANNER_SAMPLE_DIR}/raw_generations.jsonl" \
    --teacher-jsonl "${PLANNER_DATA_DIR}/train.jsonl" \
    --output-json "${NOTES_DIR}/h2_planner256_gate.json" \
    --output-md "${NOTES_DIR}/h2_planner256_gate.md" || true

PLANNER_OK="$(python - <<PY
import json
from pathlib import Path
p = json.loads(Path("${NOTES_DIR}/h2_planner256_gate.json").read_text(encoding="utf-8"))
print("1" if p.get("passed_acceptable") else "0")
PY
)"

if [ "${PLANNER_OK}" != "1" ] && [ "${RUN_DLM_IF_PLANNER_FAIL}" != "1" ]; then
  echo "H2 planner gate failed acceptable; stopping before DLM proposal." | tee -a "${LOG_DIR}/h2_decision.log"
else
  H2_SAMPLE_DIR="${OUT_DIR}/h2_plaintext_dlm_sample256"
  next_port
  run_logged "${LOG_DIR}/h2_plaintext_dlm_sample256.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_h2_plaintext_dlm.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${OUT_DIR}/h2_plaintext_dlm_sft/final" \
      --prompt-jsonl "${PLANNER_SAMPLE_DIR}/plans_for_dlm.jsonl" \
      --output-dir "${H2_SAMPLE_DIR}" \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --gen-length "${H2_GEN_LENGTH}" \
      --block-length "${H2_BLOCK_LENGTH}" \
      --temperature "${H2_TEMPERATURE}"

  run_logged "${LOG_DIR}/h2_plaintext_gate256.log" \
    python scripts/evaluate_h2_plaintext_gate.py \
      --planner-gate-json "${NOTES_DIR}/h2_planner256_gate.json" \
      --h2-sample-metrics "${H2_SAMPLE_DIR}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/h2_plaintext_gate256.json" \
      --output-md "${NOTES_DIR}/h2_plaintext_gate256.md" || true

  H2_RAW_OK="$(python - <<PY
import json
from pathlib import Path
p = json.loads(Path("${NOTES_DIR}/h2_plaintext_gate256.json").read_text(encoding="utf-8"))
print("1" if p.get("passed_acceptable") else "0")
PY
)"
  if [ "${RUN_REFINE}" = "1" ] && [ "${H2_RAW_OK}" = "1" ] && [ -f "${H2_SAMPLE_DIR}/proposal_graphs.pt" ]; then
    REFINED_DIR="${OUT_DIR}/h2_refined256"
    next_port
    run_logged "${LOG_DIR}/h2_refine256.log" \
      torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${H2_SAMPLE_DIR}/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${REFINED_DIR}" \
        --max-proposals "${SAMPLE_COUNT}" \
        --diff-steps "${DIFF_STEPS}"
    run_logged "${LOG_DIR}/h2_crysllmgen_metrics256.log" \
      python scripts/run_crysllmgen_metrics.py \
        --root-path "${REFINED_DIR}" \
        --output-json "${NOTES_DIR}/h2_crysllmgen_metrics256.json"
    REFINED_PT="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt in ${REFINED_DIR}")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
    run_logged "${LOG_DIR}/h2_composition256.log" \
      python scripts/analyze_composition_validity.py \
        --raw-pt "${H2_SAMPLE_DIR}/raw_dlm_samples.pt" \
        --raw-generations-jsonl "${H2_SAMPLE_DIR}/raw_generations.jsonl" \
        --text-key text \
        --refined-pt "${REFINED_PT}" \
        --representation crysllmgen_text \
        --refined-world-size "${REFINE_NPROC}" \
        --output-json "${NOTES_DIR}/h2_composition256.json" \
        --output-md "${NOTES_DIR}/h2_composition256.md"
    run_logged "${LOG_DIR}/h2_refined_gate256.log" \
      python scripts/evaluate_r5b_gate.py \
        --mode refined1000 \
        --sample-metrics "${H2_SAMPLE_DIR}/sample_metrics.json" \
        --composition-summary "${NOTES_DIR}/h2_composition256.json" \
        --composition-key refined_pt \
        --crysllmgen-metrics "${NOTES_DIR}/h2_crysllmgen_metrics256.json" \
        --min-graph-acceptance 0.75 \
        --min-comp-valid 0.85 \
        --min-crys-comp-valid 85.0 \
        --min-crys-struct-valid 94.0 \
        --min-crys-cov-recall 85.0 \
        --output-json "${NOTES_DIR}/h2_refined256_gate.json" || true
  fi
fi

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" python - <<'PY'
import json
import os
from pathlib import Path
notes = Path(os.environ["NOTES_DIR"])
run_id = os.environ["RUN_ID"]
def read(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
payload = {
    "run_id": run_id,
    "planner_gate": read("h2_planner256_gate.json"),
    "raw_gate": read("h2_plaintext_gate256.json"),
    "refined_gate": read("h2_refined256_gate.json"),
    "crysllmgen_metrics256": read("h2_crysllmgen_metrics256.json"),
    "composition256": read("h2_composition256.json"),
}
(notes / "h2_result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = ["# H2 LLM-Plan + Plain-Text DLM Proposal Report", "", f"- RUN_ID: {run_id}", ""]
lines += ["## Summary JSON", "", "```json", json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), "```", ""]
(notes / "h2_llm_plan_plaintext_dlm_proposal_report.md").write_text("\n".join(lines), encoding="utf-8")
PY

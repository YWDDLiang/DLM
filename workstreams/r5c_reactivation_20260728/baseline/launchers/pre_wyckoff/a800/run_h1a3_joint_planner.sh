#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1a3_joint_planner_noid_l3base}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
EPOCHS="${EPOCHS:-3}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-256}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.9}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-96}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_LR="${SFT_LR:-2e-5}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"

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
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((27000 + (${SLURM_JOB_ID:-0} % 14000)))}"
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
  "stage": "h1a3_joint_planner",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
  "data_dir": "${DATA_DIR}",
  "sample_count": int("${SAMPLE_COUNT}"),
  "epochs": int("${EPOCHS}"),
  "mixture": {
    "direct_plan": 1.0,
    "correct_plan": 0.5,
    "consistency_explain": 0.25,
    "formula_count_check": 0.25
  },
  "de_novo_constraints": {
    "gold_plan_at_sampling": False,
    "candidate_pool": False,
    "sampling_filter_or_topk": False,
    "sampling_time_repair": False
  }
}
Path("${NOTES_DIR}/h1a3_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h1_llm_planner.py \
    scripts/build_h1_llm_formula_sft_data.py \
    scripts/llama_formula_sft.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_h1_hybrid_gate.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h1_llm_planner tests.test_r5_plan_body tests.test_crysllmgen_text'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1a3_joint_planner_data.log" \
    python scripts/build_h1_llm_formula_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${PLANNER_MODEL_PATH}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id \
      --mixture direct_plan,correct_plan,consistency_explain,formula_count_check \
      --direct-plan-weight 1.0 \
      --correct-plan-weight 0.5 \
      --consistency-explain-weight 0.25 \
      --formula-count-check-weight 0.25
fi

previous_checkpoint=""
for epoch in $(seq 1 "${EPOCHS}"); do
  epoch_label="epoch${epoch}"
  train_dir="${OUT_DIR}/h1a3_${epoch_label}_llama_rich_sft"
  planner_dir="${OUT_DIR}/h1a3_${epoch_label}_planner${SAMPLE_COUNT}"
  body_dir="${OUT_DIR}/h1a3_${epoch_label}_hybrid_body${SAMPLE_COUNT}"
  refined_dir="${OUT_DIR}/h1a3_${epoch_label}_refined${REFINE_MAX_PROPOSALS}"

  train_cmd=(python scripts/llama_formula_sft.py
    --model-path "${PLANNER_MODEL_PATH}"
    --data-dir "${DATA_DIR}"
    --output-dir "${train_dir}"
    --max-length "${SFT_MAX_LENGTH}"
    --epochs 1.0
    --batch-size "${SFT_BATCH_SIZE}"
    --grad-accum "${SFT_GRAD_ACCUM}"
    --lr "${SFT_LR}")
  if [ -n "${previous_checkpoint}" ]; then
    train_cmd+=(--checkpoint-path "${previous_checkpoint}")
  fi
  run_logged "${LOG_DIR}/h1a3_${epoch_label}_llama_rich_sft.log" "${train_cmd[@]}"

  checkpoint="${train_dir}/final"
  test -d "${checkpoint}"
  previous_checkpoint="${checkpoint}"

  next_port
  run_logged "${LOG_DIR}/h1a3_${epoch_label}_planner_sample.log" \
    torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
      --model-path "${PLANNER_MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${planner_dir}" \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${PLANNER_BATCH_SIZE}" \
      --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
      --temperature "${PLANNER_TEMPERATURE}" \
      --top-p "${PLANNER_TOP_P}" \
      --top-k "${PLANNER_TOP_K}" \
      --seed "${PLANNER_SEED}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id

  run_logged "${LOG_DIR}/h1a3_${epoch_label}_planner_gate.log" \
    python scripts/evaluate_h1_planner_gate.py \
      --sample-metrics "${planner_dir}/sample_metrics.json" \
      --raw-generations-jsonl "${planner_dir}/raw_generations.jsonl" \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --output-json "${NOTES_DIR}/h1a3_${epoch_label}_planner_gate.json" \
      --output-md "${NOTES_DIR}/h1a3_${epoch_label}_planner_gate.md" || true

  plan_count="$(python - <<PY
from pathlib import Path
path = Path("${planner_dir}/plans_for_dlm.jsonl")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
)"

  next_port
  run_logged "${LOG_DIR}/h1a3_${epoch_label}_hybrid_body.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_CHECKPOINT_PATH}" \
      --prompt-jsonl "${planner_dir}/plans_for_dlm.jsonl" \
      --output-dir "${body_dir}" \
      --body-prompt-style full_plan_state \
      --num-samples "${plan_count}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --temperature "${DLM_TEMPERATURE}" \
      --freeze-plan-composition \
      --duplicate-coordinate-mask \
      --lattice-volume-mask

  run_logged "${LOG_DIR}/h1a3_${epoch_label}_hybrid_gate.log" \
    python scripts/evaluate_h1_hybrid_gate.py \
      --planner-gate-json "${NOTES_DIR}/h1a3_${epoch_label}_planner_gate.json" \
      --body-sample-metrics "${body_dir}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/h1a3_${epoch_label}_hybrid_gate.json" \
      --output-md "${NOTES_DIR}/h1a3_${epoch_label}_hybrid_gate.md" || true

  next_port
  run_logged "${LOG_DIR}/h1a3_${epoch_label}_refine.log" \
    torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
      --proposal-graphs "${body_dir}/proposal_graphs.pt" \
      --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
      --output-dir "${refined_dir}" \
      --max-proposals "${REFINE_MAX_PROPOSALS}" \
      --diff-steps "${DIFF_STEPS}"

  run_logged "${LOG_DIR}/h1a3_${epoch_label}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/h1a3_${epoch_label}_crysllmgen_metrics.json"

  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"

  run_logged "${LOG_DIR}/h1a3_${epoch_label}_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${body_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${body_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/h1a3_${epoch_label}_composition.json" \
      --output-md "${NOTES_DIR}/h1a3_${epoch_label}_composition.md"

  run_logged "${LOG_DIR}/h1a3_${epoch_label}_refined_gate.log" \
    python scripts/evaluate_r5b_gate.py \
      --mode refined1000 \
      --sample-metrics "${body_dir}/sample_metrics.json" \
      --composition-summary "${NOTES_DIR}/h1a3_${epoch_label}_composition.json" \
      --composition-key refined_pt \
      --crysllmgen-metrics "${NOTES_DIR}/h1a3_${epoch_label}_crysllmgen_metrics.json" \
      --output-json "${NOTES_DIR}/h1a3_${epoch_label}_refined_gate.json" || true
done

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
summary = {"run_id": "${RUN_ID}", "epochs": []}
for path in sorted(notes.glob("h1a3_epoch*_refined_gate.json")):
    summary["epochs"].append({"gate": path.name, "payload": json.loads(path.read_text())})
(notes / "h1a3_joint_planner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

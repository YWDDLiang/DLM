#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
H1A2_PLANNER_CHECKPOINT="${H1A2_PLANNER_CHECKPOINT:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final}"
H1A2_PLANS_JSONL="${H1A2_PLANS_JSONL:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_r5_exact_length}"
GPU_COUNT="${GPU_COUNT:-2}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-256}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-96}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.9}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
POSITION_DIAGNOSTICS_STEPS="${POSITION_DIAGNOSTICS_STEPS:-200}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${PLANNER_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
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

BASE_MASTER_PORT="${MASTER_PORT:-$((28000 + (${SLURM_JOB_ID:-0} % 12000)))}"
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
  "stage": "h1_free_geometry_executor",
  "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
  "h1a2_planner_checkpoint": "${H1A2_PLANNER_CHECKPOINT}",
  "h1a2_plans_jsonl": "${H1A2_PLANS_JSONL}",
  "sample_count": int("${SAMPLE_COUNT}"),
  "loss_sweep": {
    "lattice_up": {"lambda_len": 1.5, "lambda_ang": 1.5, "lambda_coord": 1.0},
    "coord_up": {"lambda_len": 1.0, "lambda_ang": 1.0, "lambda_coord": 1.5},
    "balanced": {"lambda_len": 1.5, "lambda_ang": 1.5, "lambda_coord": 1.25}
  }
}
Path("${NOTES_DIR}/h1_free_geometry_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/llada_sft.py \
    scripts/build_r5_exact_length_sft_data.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_llada_sft_weights tests.test_llada_generation_masks tests.test_r5_dynamic_length'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_r5_exact_length_data.log" \
    python scripts/build_r5_exact_length_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${DLM_CHECKPOINT_PATH}"
fi

FIXED_PLANS="${H1A2_PLANS_JSONL}"
if [ ! -f "${FIXED_PLANS}" ]; then
  if [ ! -d "${H1A2_PLANNER_CHECKPOINT}" ]; then
    echo "Missing H1-A2 plans and planner checkpoint: ${FIXED_PLANS}, ${H1A2_PLANNER_CHECKPOINT}" >&2
    exit 2
  fi
  PLANNER_DIR="${OUT_DIR}/h1a2_epoch2_fixed_planner${SAMPLE_COUNT}"
  next_port
  run_logged "${LOG_DIR}/regenerate_h1a2_epoch2_fixed_plans.log" \
    torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
      --model-path "${PLANNER_MODEL_PATH}" \
      --checkpoint-path "${H1A2_PLANNER_CHECKPOINT}" \
      --output-dir "${PLANNER_DIR}" \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${PLANNER_BATCH_SIZE}" \
      --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
      --temperature "${PLANNER_TEMPERATURE}" \
      --top-p "${PLANNER_TOP_P}" \
      --top-k "${PLANNER_TOP_K}" \
      --seed "${PLANNER_SEED}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id
  FIXED_PLANS="${PLANNER_DIR}/plans_for_dlm.jsonl"
fi
echo "${FIXED_PLANS}" > "${NOTES_DIR}/fixed_plans_path.txt"

sample_refine_metric() {
  local label="$1"
  local checkpoint="$2"
  shift 2
  local sample_dir="${OUT_DIR}/${label}_sample${SAMPLE_COUNT}"
  local refined_dir="${OUT_DIR}/${label}_refined${REFINE_MAX_PROPOSALS}"
  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${NOTES_DIR}/${label}_crysllmgen_metrics.json" ]; then
    echo "Skipping completed sample/refine/metrics for ${label}."
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${label}_sample.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --prompt-jsonl "${FIXED_PLANS}" \
      --output-dir "${sample_dir}" \
      --body-prompt-style full_plan_state \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --temperature "${DLM_TEMPERATURE}" \
      "$@"

  run_logged "${LOG_DIR}/${label}_raw_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${sample_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --text-key text \
      --representation dynamic_v1 \
      --output-json "${NOTES_DIR}/${label}_composition_raw.json" \
      --output-md "${NOTES_DIR}/${label}_composition_raw.md"

  next_port
  run_logged "${LOG_DIR}/${label}_refine.log" \
    torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
      --proposal-graphs "${sample_dir}/proposal_graphs.pt" \
      --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
      --output-dir "${refined_dir}" \
      --max-proposals "${REFINE_MAX_PROPOSALS}" \
      --diff-steps "${DIFF_STEPS}"

  run_logged "${LOG_DIR}/${label}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/${label}_crysllmgen_metrics.json"

  local refined_pt
  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
  run_logged "${LOG_DIR}/${label}_composition_refined.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${sample_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/${label}_composition.json" \
      --output-md "${NOTES_DIR}/${label}_composition.md"
}

sample_refine_metric "ablation_default" "${DLM_CHECKPOINT_PATH}" \
  --freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan
sample_refine_metric "ablation_no_lattice_volume_mask" "${DLM_CHECKPOINT_PATH}" \
  --freeze-plan-composition --duplicate-coordinate-mask --no-lattice-volume-mask --generation-schedule exact-plan
sample_refine_metric "ablation_no_duplicate_coordinate_mask" "${DLM_CHECKPOINT_PATH}" \
  --freeze-plan-composition --no-duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan
sample_refine_metric "ablation_default_schedule" "${DLM_CHECKPOINT_PATH}" \
  --freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule default
sample_refine_metric "ablation_no_freeze_plan_composition" "${DLM_CHECKPOINT_PATH}" \
  --no-freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan

train_weighted_variant() {
  local label="$1"
  local len_w="$2"
  local ang_w="$3"
  local coord_w="$4"
  local train_dir="${OUT_DIR}/${label}_sft"
  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${NOTES_DIR}/${label}_crysllmgen_metrics.json" ]; then
    echo "Skipping completed weighted variant ${label}."
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${label}_sft.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_CHECKPOINT_PATH}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${train_dir}" \
      --representation dynamic_v1 \
      --max-length "${SFT_MAX_LENGTH}" \
      --epochs "${SFT_EPOCHS}" \
      --batch-size "${SFT_BATCH_SIZE}" \
      --grad-accum "${SFT_GRAD_ACCUM}" \
      --lr "${SFT_LR}" \
      --position-diagnostics-steps "${POSITION_DIAGNOSTICS_STEPS}" \
      --dynamic-lattice-length-loss-weight "${len_w}" \
      --dynamic-lattice-angle-loss-weight "${ang_w}" \
      --dynamic-coord-loss-weight "${coord_w}"
  sample_refine_metric "${label}" "${train_dir}/final" \
    --freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan
}

train_weighted_variant "weighted_lattice_up" 1.5 1.5 1.0
train_weighted_variant "weighted_coord_up" 1.0 1.0 1.5
train_weighted_variant "weighted_balanced" 1.5 1.5 1.25

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
summary = {"run_id": "${RUN_ID}", "labels": []}
for path in sorted(notes.glob("*_crysllmgen_metrics.json")):
    label = path.name.replace("_crysllmgen_metrics.json", "")
    item = {"label": label, "crysllmgen_metrics": json.loads(path.read_text())}
    comp = notes / f"{label}_composition.json"
    if comp.exists():
        item["composition"] = json.loads(comp.read_text())
    summary["labels"].append(item)
(notes / "h1_free_geometry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

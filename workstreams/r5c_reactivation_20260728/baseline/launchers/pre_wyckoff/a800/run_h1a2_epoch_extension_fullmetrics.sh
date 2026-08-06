#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base}"
EPOCH1_CHECKPOINT="${EPOCH1_CHECKPOINT:-runs/20260602_182700-h1a2-rich-l3base-256/outputs/h1a2_llama_rich_sft/final}"
START_EPOCH="${START_EPOCH:-2}"
END_EPOCH="${END_EPOCH:-3}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
FULL_SAMPLES="${FULL_SAMPLES:-1200}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-1000}"
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
SFT_EPOCHS_PER_STEP="${SFT_EPOCHS_PER_STEP:-1.0}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
RUN_A100_SUN="${RUN_A100_SUN:-1}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"

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

BASE_MASTER_PORT="${MASTER_PORT:-$((29000 + (${SLURM_JOB_ID:-0} % 12000)))}"
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

activate_conda_env() {
  local env_name="$1"
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${env_name}"
  set -u
}

trap 'status=$?; if [ -n "${MP_API_KEY_FILE:-}" ] && [ -f "${MP_API_KEY_FILE}" ] && [[ "${MP_API_KEY_FILE}" == /tmp/h1a2_epoch_mp_key_* ]]; then rm -f "${MP_API_KEY_FILE}"; fi; echo "${status}" > "${NOTES_DIR}/exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/end_time.txt"; nvidia-smi > "${NOTES_DIR}/gpu_status_end.txt" 2>&1 || true; exit "${status}"' EXIT

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
  "stage": "h1a2_epoch_extension_fullmetrics",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "epoch1_checkpoint": "${EPOCH1_CHECKPOINT}",
  "start_epoch": int("${START_EPOCH}"),
  "end_epoch": int("${END_EPOCH}"),
  "planner_temperature": float("${PLANNER_TEMPERATURE}"),
  "full_samples": int("${FULL_SAMPLES}"),
  "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
  "run_a100_sun": "${RUN_A100_SUN}" == "1",
  "a100_env_name": "${A100_ENV_NAME}",
  "main_env_name": "${MAIN_ENV_NAME}",
  "de_novo_constraints": {
    "gold_plan_at_sampling": False,
    "candidate_pool": False,
    "sampling_filter_or_topk": False,
    "temperature_is_fixed_method_level": True
  }
}
Path("${NOTES_DIR}/h1a2_epoch_extension_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/llama_formula_sft.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_h1_hybrid_gate.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py

test -d "${DATA_DIR}"
test -d "${EPOCH1_CHECKPOINT}"

previous_checkpoint="${EPOCH1_CHECKPOINT}"
for epoch in $(seq "${START_EPOCH}" "${END_EPOCH}"); do
  epoch_label="epoch${epoch}"
  train_dir="${OUT_DIR}/h1a2_${epoch_label}_llama_rich_sft"
  planner_dir="${OUT_DIR}/h1a2_${epoch_label}_planner${FULL_SAMPLES}"
  body_dir="${OUT_DIR}/h1a2_${epoch_label}_hybrid_body${FULL_SAMPLES}"
  refined_dir="${OUT_DIR}/h1a2_${epoch_label}_refined${REFINE_MAX_PROPOSALS}"

  if [ ! -d "${train_dir}/final" ]; then
    run_logged "${LOG_DIR}/h1a2_${epoch_label}_llama_rich_sft.log" \
      python scripts/llama_formula_sft.py \
        --model-path "${PLANNER_MODEL_PATH}" \
        --checkpoint-path "${previous_checkpoint}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${train_dir}" \
        --max-length "${SFT_MAX_LENGTH}" \
        --epochs "${SFT_EPOCHS_PER_STEP}" \
        --batch-size "${SFT_BATCH_SIZE}" \
        --grad-accum "${SFT_GRAD_ACCUM}" \
        --lr "${SFT_LR}"
  fi

  checkpoint="${train_dir}/final"
  test -d "${checkpoint}"

  next_port
  run_logged "${LOG_DIR}/h1a2_${epoch_label}_planner_sample${FULL_SAMPLES}.log" \
    torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
      --model-path "${PLANNER_MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${planner_dir}" \
      --num-samples "${FULL_SAMPLES}" \
      --batch-size "${PLANNER_BATCH_SIZE}" \
      --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
      --temperature "${PLANNER_TEMPERATURE}" \
      --top-p "${PLANNER_TOP_P}" \
      --top-k "${PLANNER_TOP_K}" \
      --seed "${PLANNER_SEED}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id

  run_logged "${LOG_DIR}/h1a2_${epoch_label}_planner_gate${FULL_SAMPLES}.log" \
    python scripts/evaluate_h1_planner_gate.py \
      --sample-metrics "${planner_dir}/sample_metrics.json" \
      --raw-generations-jsonl "${planner_dir}/raw_generations.jsonl" \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --output-json "${NOTES_DIR}/h1a2_${epoch_label}_planner_gate.json" \
      --output-md "${NOTES_DIR}/h1a2_${epoch_label}_planner_gate.md" || true

  plan_count="$(python - <<PY
from pathlib import Path
path = Path("${planner_dir}/plans_for_dlm.jsonl")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
)"

  next_port
  run_logged "${LOG_DIR}/h1a2_${epoch_label}_hybrid_body.log" \
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

  run_logged "${LOG_DIR}/h1a2_${epoch_label}_hybrid_gate.log" \
    python scripts/evaluate_h1_hybrid_gate.py \
      --planner-gate-json "${NOTES_DIR}/h1a2_${epoch_label}_planner_gate.json" \
      --body-sample-metrics "${body_dir}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/h1a2_${epoch_label}_hybrid_gate.json" \
      --output-md "${NOTES_DIR}/h1a2_${epoch_label}_hybrid_gate.md" || true

  next_port
  run_logged "${LOG_DIR}/h1a2_${epoch_label}_refine.log" \
    torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
      --proposal-graphs "${body_dir}/proposal_graphs.pt" \
      --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
      --output-dir "${refined_dir}" \
      --max-proposals "${REFINE_MAX_PROPOSALS}" \
      --diff-steps "${DIFF_STEPS}"

  run_logged "${LOG_DIR}/h1a2_${epoch_label}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/h1a2_${epoch_label}_crysllmgen_metrics.json"

  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt in ${refined_dir}")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"

  run_logged "${LOG_DIR}/h1a2_${epoch_label}_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${body_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${body_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/h1a2_${epoch_label}_composition.json" \
      --output-md "${NOTES_DIR}/h1a2_${epoch_label}_composition.md"

  run_logged "${LOG_DIR}/h1a2_${epoch_label}_refined_gate.log" \
    python scripts/evaluate_r5b_gate.py \
      --mode refined1000 \
      --sample-metrics "${body_dir}/sample_metrics.json" \
      --composition-summary "${NOTES_DIR}/h1a2_${epoch_label}_composition.json" \
      --composition-key refined_pt \
      --crysllmgen-metrics "${NOTES_DIR}/h1a2_${epoch_label}_crysllmgen_metrics.json" \
      --min-comp-valid 0.85 \
      --min-crys-comp-valid 85.0 \
      --min-crys-struct-valid 94.0 \
      --min-crys-cov-recall 85.0 \
      --output-json "${NOTES_DIR}/h1a2_${epoch_label}_refined_gate.json" || true

  if [ "${RUN_A100_SUN}" = "1" ]; then
    activate_conda_env "${A100_ENV_NAME}"
    a100_run_id="${RUN_ID}-a100-${epoch_label}"
    a100_notes="runs/${a100_run_id}/notes"
    mkdir -p "${a100_notes}"

    run_logged "${LOG_DIR}/h1a2_${epoch_label}_a100_cache_missing_pre.log" \
      python scripts/a800/check_a100_eval_sun_cache_missing.py \
        --eval-dir reference/a100_eval_sun \
        --train-csv reference/crysllmgen/data/mp_20/train.csv \
        --cache-path "${MP_CACHE_PATH}" \
        --run "dlm=${refined_pt}" \
        --summary-json "${NOTES_DIR}/h1a2_${epoch_label}_a100_cache_missing_pre.json"

    missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/h1a2_${epoch_label}_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
    if [ "${missing_count}" != "0" ]; then
      if [ -z "${MP_API_KEY_FILE}" ] || [ ! -s "${MP_API_KEY_FILE}" ]; then
        echo "A100 MP cache missing entries for ${epoch_label}, but MP_API_KEY_FILE is empty or missing; skipping A100 S.U.N. for this epoch." >&2
        activate_conda_env "${MAIN_ENV_NAME}"
        continue
      fi
      run_logged "${LOG_DIR}/h1a2_${epoch_label}_a100_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${refined_pt}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/h1a2_${epoch_label}_a100_cache_enrich.json" || {
            echo "A100 MP cache enrichment failed for ${epoch_label}; skipping A100 S.U.N. for this epoch." >&2
            activate_conda_env "${MAIN_ENV_NAME}"
            continue
          }
    fi

    run_logged "${LOG_DIR}/h1a2_${epoch_label}_a100_sun.log" \
      env RUN_ID="${a100_run_id}" DLM_PT="${refined_pt}" PYTHON_BIN=python \
        MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
        bash scripts/a800/run_a100_eval_sun_dlm_only.sh || {
          echo "A100 S.U.N. failed for ${epoch_label}; continuing remaining epochs." >&2
          activate_conda_env "${MAIN_ENV_NAME}"
          continue
        }
    cp "${a100_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/h1a2_${epoch_label}_a100_sun_summary.json"
    cp "${a100_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/h1a2_${epoch_label}_a100_sun_strict_summary.md"
    cp "${a100_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/h1a2_${epoch_label}_a100_sun_meta_like_summary.md"
    activate_conda_env "${MAIN_ENV_NAME}"
  fi

  previous_checkpoint="${checkpoint}"
done

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" START_EPOCH="${START_EPOCH}" END_EPOCH="${END_EPOCH}" python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])
summary = {"run_id": os.environ["RUN_ID"], "epochs": {}}
for epoch in range(int(os.environ["START_EPOCH"]), int(os.environ["END_EPOCH"]) + 1):
    label = f"epoch{epoch}"
    item = {}
    for name in ("planner_gate", "hybrid_gate", "refined_gate", "crysllmgen_metrics", "composition", "a100_sun_summary"):
        path = notes / f"h1a2_{label}_{name}.json"
        if path.exists():
            item[name] = json.loads(path.read_text(encoding="utf-8"))
    summary["epochs"][label] = item
notes.joinpath("h1a2_epoch_extension_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# H1-A2 Epoch Extension Full Metrics", "", f"- RUN_ID: {os.environ['RUN_ID']}", ""]
for label, item in summary["epochs"].items():
    metrics = ((item.get("crysllmgen_metrics") or {}).get("metrics") or {})
    a100 = item.get("a100_sun_summary") or {}
    strict = (a100.get("dlm_strict") or {}).get("coverage-adjusted_sun_estimate_pct")
    meta = (a100.get("dlm_meta_like") or {}).get("coverage-adjusted_sun_estimate_pct")
    lines.extend([
        f"## {label}",
        "",
        f"- comp_valid: `{metrics.get('comp_valid')}`",
        f"- struct_valid: `{metrics.get('struct_valid')}`",
        f"- cov_recall: `{metrics.get('cov_recall')}`",
        f"- cov_precision: `{metrics.get('cov_precision')}`",
        f"- A100 strict adjusted: `{strict}`",
        f"- A100 meta-like adjusted: `{meta}`",
        "",
    ])
notes.joinpath("h1a2_epoch_extension_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

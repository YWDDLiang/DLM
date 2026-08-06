#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_WARM_START="${DLM_WARM_START:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
DLM_TOKENIZER_PATH="${DLM_TOKENIZER_PATH:-${DLM_WARM_START}}"
PLANS_JSONL="${PLANS_JSONL:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1g1_robust_exact_dlm}"
GPU_COUNT="${GPU_COUNT:-2}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1200}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-1000}"
CONDITION_VIEWS="${CONDITION_VIEWS:-full-rich,condition-dropout,formula-volume-sg,formula-volume-only}"
TRAIN_FINAL_OVERRIDE="${TRAIN_FINAL_OVERRIDE:-}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
POSITION_DIAGNOSTICS_STEPS="${POSITION_DIAGNOSTICS_STEPS:-200}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
RUN_A100_SUN="${RUN_A100_SUN:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
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

read_condition_views() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  CONDITION_VIEW_ARRAY=(${raw})
  if [ "${#CONDITION_VIEW_ARRAY[@]}" -eq 0 ]; then
    echo "No condition views configured." >&2
    exit 2
  fi
  CONDITION_VIEWS_NORMALIZED="${CONDITION_VIEW_ARRAY[*]}"
}

read_condition_views "${CONDITION_VIEWS}"

BASE_MASTER_PORT="${MASTER_PORT:-$((32000 + (${SLURM_JOB_ID:-0} % 8000)))}"
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

cleanup_key() {
  case "${MP_API_KEY_FILE:-}" in
    /tmp/h1g1_mp_key_*|/public/home/jiaosz/.cache/codex/h1g1_mp_key_*)
      [ -f "${MP_API_KEY_FILE}" ] && rm -f "${MP_API_KEY_FILE}"
      ;;
  esac
}

trap 'status=$?; cleanup_key; echo "${status}" > "${NOTES_DIR}/exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/end_time.txt"; nvidia-smi > "${NOTES_DIR}/gpu_status_end.txt" 2>&1 || true; exit "${status}"' EXIT

date "+%F %T %Z" > "${NOTES_DIR}/start_time.txt"
{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-none}"
  echo "condition_views=${CONDITION_VIEWS_NORMALIZED}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort | grep -v -i 'api\|key\|token\|secret' > "${NOTES_DIR}/environment_redacted.txt"

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "h1g1_robust_exact_dlm",
  "dlm_warm_start": "${DLM_WARM_START}",
  "plans_jsonl": "${PLANS_JSONL}",
  "data_dir": "${DATA_DIR}",
  "condition_views": "${CONDITION_VIEWS_NORMALIZED}".split(),
  "sample_count": int("${SAMPLE_COUNT}"),
  "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
  "loss_proxy": {
    "implementation": "condition-view prompted geometry CE with dynamic lattice/coord group diagnostics",
    "lambda_free": 1.15,
    "lambda_volume": "prompted volume-bin consistency",
    "lambda_density": "teacher real-structure density/volume basin via body CE"
  },
  "a100_baseline_reference_not_rerun": {
    "strict_adjusted_pct": 9.31,
    "meta_like_adjusted_pct": 47.67,
    "r5c_conditional_strict_adjusted_pct": 10.61,
    "r5c_conditional_meta_like_adjusted_pct": 74.38
  }
}
Path("${NOTES_DIR}/h1g1_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

activate_conda_env "${MAIN_ENV_NAME}"

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h1_formula_only_body.py \
    scripts/build_h1g1_robust_exact_dlm_sft_data.py \
    scripts/build_h1g1_condition_view_prompts.py \
    scripts/llada_sft.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h1_formula_only_body tests.test_llada_sft_weights tests.test_r5_dynamic_length'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1g1_robust_exact_dlm_data.log" \
    python scripts/build_h1g1_robust_exact_dlm_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${DLM_TOKENIZER_PATH}" \
      --condition-views full-rich,condition-dropout,formula-volume-sg,formula-volume-only \
      --full-rich-weight 1.0 \
      --condition-dropout-weight 0.5 \
      --formula-volume-sg-weight 0.5 \
      --formula-volume-only-weight 0.25
fi

TRAIN_DIR="${OUT_DIR}/h1g1_robust_exact_dlm_sft"
if [ -z "${TRAIN_FINAL_OVERRIDE}" ] && { [ "${SKIP_COMPLETED}" != "1" ] || [ ! -d "${TRAIN_DIR}/final" ]; }; then
  next_port
  run_logged "${LOG_DIR}/h1g1_robust_exact_dlm_sft.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_WARM_START}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${TRAIN_DIR}" \
      --representation dynamic_v1 \
      --max-length "${SFT_MAX_LENGTH}" \
      --epochs "${SFT_EPOCHS}" \
      --batch-size "${SFT_BATCH_SIZE}" \
      --grad-accum "${SFT_GRAD_ACCUM}" \
      --lr "${SFT_LR}" \
      --position-diagnostics-steps "${POSITION_DIAGNOSTICS_STEPS}" \
      --dynamic-lattice-length-loss-weight 1.15 \
      --dynamic-lattice-angle-loss-weight 1.10 \
      --dynamic-coord-loss-weight 1.15
fi
EVAL_CKPT="${TRAIN_FINAL_OVERRIDE:-${TRAIN_DIR}/final}"
test -d "${EVAL_CKPT}"
test -f "${PLANS_JSONL}"

run_a100_for_refined() {
  local label="$1"
  local refined_pt="$2"
  if [ "${RUN_A100_SUN}" != "1" ]; then
    return 0
  fi
  activate_conda_env "${A100_ENV_NAME}"
  local a100_run_id="${RUN_ID}-a100-${label}"
  local a100_notes="runs/${a100_run_id}/notes"
  mkdir -p "${a100_notes}"
  run_logged "${LOG_DIR}/${label}_a100_cache_missing_pre.log" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
      --eval-dir reference/a100_eval_sun \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --run "dlm=${refined_pt}" \
      --summary-json "${NOTES_DIR}/${label}_a100_cache_missing_pre.json"
  local missing_count
  missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/${label}_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
  if [ "${missing_count}" != "0" ]; then
    if [ -n "${MP_API_KEY_FILE}" ] && [ -s "${MP_API_KEY_FILE}" ]; then
      run_logged "${LOG_DIR}/${label}_a100_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${refined_pt}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/${label}_a100_cache_enrich.json"
    elif [ "${ALLOW_MISSING_CACHE}" != "1" ]; then
      echo "A100 cache missing ${missing_count} entries for ${label}; provide MP_API_KEY_FILE or set ALLOW_MISSING_CACHE=1." >&2
      exit 2
    fi
  fi
  run_logged "${LOG_DIR}/${label}_a100_sun.log" \
    env RUN_ID="${a100_run_id}" DLM_PT="${refined_pt}" PYTHON_BIN=python \
      MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
      ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
      bash scripts/a800/run_a100_eval_sun_dlm_only.sh
  cp "${a100_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/${label}_a100_sun_summary.json"
  cp "${a100_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/${label}_a100_sun_strict_summary.md"
  cp "${a100_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/${label}_a100_sun_meta_like_summary.md"
  activate_conda_env "${MAIN_ENV_NAME}"
}

sample_refine_metric() {
  local view="$1"
  local label="h1g1_${view//-/_}"
  local prompt_jsonl="${OUT_DIR}/${label}_plans.jsonl"
  local sample_dir="${OUT_DIR}/${label}_sample${SAMPLE_COUNT}"
  local refined_dir="${OUT_DIR}/${label}_refined${REFINE_MAX_PROPOSALS}"

  run_logged "${LOG_DIR}/${label}_build_condition_prompts.log" \
    python scripts/build_h1g1_condition_view_prompts.py \
      --input-jsonl "${PLANS_JSONL}" \
      --output-jsonl "${prompt_jsonl}" \
      --condition-view "${view}" \
      --limit "${SAMPLE_COUNT}"

  if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${sample_dir}/sample_metrics.json" ]; then
    next_port
    run_logged "${LOG_DIR}/${label}_sample${SAMPLE_COUNT}.log" \
      torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
        --model-path "${DLM_MODEL_PATH}" \
        --checkpoint-path "${EVAL_CKPT}" \
        --prompt-jsonl "${prompt_jsonl}" \
        --output-dir "${sample_dir}" \
        --body-prompt-style full_plan_state \
        --num-samples "${SAMPLE_COUNT}" \
        --batch-size "${DLM_BATCH_SIZE}" \
        --temperature "${DLM_TEMPERATURE}" \
        --freeze-plan-composition \
        --duplicate-coordinate-mask \
        --lattice-volume-mask \
        --generation-schedule exact-plan
  fi

  if [ "${SKIP_COMPLETED}" != "1" ] || ! find "${refined_dir}" -maxdepth 1 -type f -name 'dlm_refined_mp_*.pt' ! -name '*.rank*.pt' | grep -q .; then
    next_port
    run_logged "${LOG_DIR}/${label}_refine${REFINE_MAX_PROPOSALS}.log" \
      torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${sample_dir}/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${refined_dir}" \
        --max-proposals "${REFINE_MAX_PROPOSALS}" \
        --diff-steps "${DIFF_STEPS}"
  fi

  local refined_pt
  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
  echo "${refined_pt}" > "${NOTES_DIR}/${label}_refined_pt.txt"

  run_logged "${LOG_DIR}/${label}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/${label}_crysllmgen_metrics.json"

  run_logged "${LOG_DIR}/${label}_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${sample_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/${label}_composition.json" \
      --output-md "${NOTES_DIR}/${label}_composition.md"

  run_logged "${LOG_DIR}/${label}_refined_gate.log" \
    python scripts/evaluate_r5b_gate.py \
      --mode refined1000 \
      --sample-metrics "${sample_dir}/sample_metrics.json" \
      --composition-summary "${NOTES_DIR}/${label}_composition.json" \
      --composition-key refined_pt \
      --crysllmgen-metrics "${NOTES_DIR}/${label}_crysllmgen_metrics.json" \
      --min-comp-valid 0.85 \
      --min-crys-comp-valid 85.0 \
      --min-crys-struct-valid 94.0 \
      --min-crys-cov-recall 85.0 \
      --output-json "${NOTES_DIR}/${label}_refined_gate.json" || true

  run_a100_for_refined "${label}" "${refined_pt}"
}

for view in "${CONDITION_VIEW_ARRAY[@]}"; do
  sample_refine_metric "${view}"
done

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" CONDITION_VIEWS="${CONDITION_VIEWS_NORMALIZED}" python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])
summary = {"run_id": os.environ["RUN_ID"], "views": {}}
for view in os.environ["CONDITION_VIEWS"].split():
    label = "h1g1_" + view.replace("-", "_")
    item = {}
    for name in ("refined_gate", "crysllmgen_metrics", "composition", "a100_sun_summary"):
        path = notes / f"{label}_{name}.json"
        if path.exists():
            item[name] = json.loads(path.read_text(encoding="utf-8"))
    pt = notes / f"{label}_refined_pt.txt"
    if pt.exists():
        item["refined_pt"] = pt.read_text(encoding="utf-8").strip()
    summary["views"][view] = item
notes.joinpath("h1g1_robust_exact_dlm_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# H1-G1 Robust Exact-DLM Report", "", f"- RUN_ID: `{os.environ['RUN_ID']}`", ""]
for view, item in summary["views"].items():
    crys = ((item.get("crysllmgen_metrics") or {}).get("metrics") or {})
    a100 = item.get("a100_sun_summary") or {}
    strict = (a100.get("dlm_strict") or {}).get("coverage-adjusted_sun_estimate_pct")
    meta = (a100.get("dlm_meta_like") or {}).get("coverage-adjusted_sun_estimate_pct")
    lines.extend([
        f"## {view}",
        "",
        f"- comp_valid: `{crys.get('comp_valid')}`",
        f"- struct_valid: `{crys.get('struct_valid')}`",
        f"- cov_recall: `{crys.get('cov_recall')}`",
        f"- density WDist: `{crys.get('wdist_density')}`",
        f"- A100 strict adjusted: `{strict}`",
        f"- A100 meta-like adjusted: `{meta}`",
        "",
    ])
notes.joinpath("h1g1_robust_exact_dlm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

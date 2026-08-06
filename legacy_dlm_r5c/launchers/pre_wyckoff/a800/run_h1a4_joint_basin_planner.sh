#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1a4_joint_basin_planner}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
EPOCHS="${EPOCHS:-2}"
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
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
RUN_A100_SUN="${RUN_A100_SUN:-1}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

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

BASE_MASTER_PORT="${MASTER_PORT:-$((31000 + (${SLURM_JOB_ID:-0} % 9000)))}"
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
    /tmp/h1a4_mp_key_*|/public/home/jiaosz/.cache/codex/h1a4_mp_key_*)
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
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort | grep -v -i 'api\|key\|token\|secret' > "${NOTES_DIR}/environment_redacted.txt"

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "h1a4_joint_basin_planner",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
  "data_dir": "${DATA_DIR}",
  "epochs": int("${EPOCHS}"),
  "full_samples": int("${FULL_SAMPLES}"),
  "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
  "mixture": {
    "direct_plan": 1.0,
    "correct_plan": 0.5,
    "consistency_explain": 0.25,
    "formula_count_check": 0.25
  },
  "no_sampling_filter": True,
  "a100_baseline_reference_not_rerun": {
    "strict_adjusted_pct": 9.31,
    "meta_like_adjusted_pct": 47.67,
    "r5c_conditional_strict_adjusted_pct": 10.61,
    "r5c_conditional_meta_like_adjusted_pct": 74.38
  }
}
Path("${NOTES_DIR}/h1a4_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

activate_conda_env "${MAIN_ENV_NAME}"

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
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h1_llm_planner tests.test_r5_plan_body tests.test_crysllmgen_text'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1a4_joint_basin_data.log" \
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

run_a100_for_refined() {
  local epoch_label="$1"
  local refined_pt="$2"
  if [ "${RUN_A100_SUN}" != "1" ]; then
    return 0
  fi
  activate_conda_env "${A100_ENV_NAME}"
  local a100_run_id="${RUN_ID}-a100-${epoch_label}"
  local a100_notes="runs/${a100_run_id}/notes"
  mkdir -p "${a100_notes}"
  run_logged "${LOG_DIR}/h1a4_${epoch_label}_a100_cache_missing_pre.log" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
      --eval-dir reference/a100_eval_sun \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --run "dlm=${refined_pt}" \
      --summary-json "${NOTES_DIR}/h1a4_${epoch_label}_a100_cache_missing_pre.json"
  local missing_count
  missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/h1a4_${epoch_label}_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
  if [ "${missing_count}" != "0" ]; then
    if [ -n "${MP_API_KEY_FILE}" ] && [ -s "${MP_API_KEY_FILE}" ]; then
      run_logged "${LOG_DIR}/h1a4_${epoch_label}_a100_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${refined_pt}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/h1a4_${epoch_label}_a100_cache_enrich.json"
    elif [ "${ALLOW_MISSING_CACHE}" != "1" ]; then
      echo "A100 cache missing ${missing_count} entries for ${epoch_label}; provide MP_API_KEY_FILE or set ALLOW_MISSING_CACHE=1." >&2
      exit 2
    fi
  fi
  run_logged "${LOG_DIR}/h1a4_${epoch_label}_a100_sun.log" \
    env RUN_ID="${a100_run_id}" DLM_PT="${refined_pt}" PYTHON_BIN=python \
      MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
      ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
      bash scripts/a800/run_a100_eval_sun_dlm_only.sh
  cp "${a100_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/h1a4_${epoch_label}_a100_sun_summary.json"
  cp "${a100_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/h1a4_${epoch_label}_a100_sun_strict_summary.md"
  cp "${a100_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/h1a4_${epoch_label}_a100_sun_meta_like_summary.md"
  activate_conda_env "${MAIN_ENV_NAME}"
}

previous_checkpoint=""
for epoch in $(seq 1 "${EPOCHS}"); do
  epoch_label="epoch${epoch}"
  train_dir="${OUT_DIR}/h1a4_${epoch_label}_llama_rich_sft"
  planner_dir="${OUT_DIR}/h1a4_${epoch_label}_planner${FULL_SAMPLES}"
  body_dir="${OUT_DIR}/h1a4_${epoch_label}_hybrid_body${FULL_SAMPLES}"
  refined_dir="${OUT_DIR}/h1a4_${epoch_label}_refined${REFINE_MAX_PROPOSALS}"

  if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -d "${train_dir}/final" ]; then
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
    run_logged "${LOG_DIR}/h1a4_${epoch_label}_llama_rich_sft.log" "${train_cmd[@]}"
  fi
  checkpoint="${train_dir}/final"
  test -d "${checkpoint}"
  previous_checkpoint="${checkpoint}"

  if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${planner_dir}/sample_metrics.json" ]; then
    next_port
    run_logged "${LOG_DIR}/h1a4_${epoch_label}_planner_sample${FULL_SAMPLES}.log" \
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
  fi

  run_logged "${LOG_DIR}/h1a4_${epoch_label}_planner_gate.log" \
    python scripts/evaluate_h1_planner_gate.py \
      --sample-metrics "${planner_dir}/sample_metrics.json" \
      --raw-generations-jsonl "${planner_dir}/raw_generations.jsonl" \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --output-json "${NOTES_DIR}/h1a4_${epoch_label}_planner_gate.json" \
      --output-md "${NOTES_DIR}/h1a4_${epoch_label}_planner_gate.md" || true

  plan_count="$(python - <<PY
from pathlib import Path
print(sum(1 for line in Path("${planner_dir}/plans_for_dlm.jsonl").open(encoding="utf-8") if line.strip()))
PY
)"

  if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${body_dir}/sample_metrics.json" ]; then
    next_port
    run_logged "${LOG_DIR}/h1a4_${epoch_label}_hybrid_body.log" \
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
        --lattice-volume-mask \
        --generation-schedule exact-plan
  fi

  run_logged "${LOG_DIR}/h1a4_${epoch_label}_hybrid_gate.log" \
    python scripts/evaluate_h1_hybrid_gate.py \
      --planner-gate-json "${NOTES_DIR}/h1a4_${epoch_label}_planner_gate.json" \
      --body-sample-metrics "${body_dir}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/h1a4_${epoch_label}_hybrid_gate.json" \
      --output-md "${NOTES_DIR}/h1a4_${epoch_label}_hybrid_gate.md" || true

  if [ "${SKIP_COMPLETED}" != "1" ] || ! find "${refined_dir}" -maxdepth 1 -type f -name 'dlm_refined_mp_*.pt' ! -name '*.rank*.pt' | grep -q .; then
    next_port
    run_logged "${LOG_DIR}/h1a4_${epoch_label}_refine${REFINE_MAX_PROPOSALS}.log" \
      torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${body_dir}/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${refined_dir}" \
        --max-proposals "${REFINE_MAX_PROPOSALS}" \
        --diff-steps "${DIFF_STEPS}"
  fi

  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
  echo "${refined_pt}" > "${NOTES_DIR}/h1a4_${epoch_label}_refined_pt.txt"

  run_logged "${LOG_DIR}/h1a4_${epoch_label}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/h1a4_${epoch_label}_crysllmgen_metrics.json"

  run_logged "${LOG_DIR}/h1a4_${epoch_label}_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${body_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${body_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/h1a4_${epoch_label}_composition.json" \
      --output-md "${NOTES_DIR}/h1a4_${epoch_label}_composition.md"

  run_logged "${LOG_DIR}/h1a4_${epoch_label}_refined_gate.log" \
    python scripts/evaluate_r5b_gate.py \
      --mode refined1000 \
      --sample-metrics "${body_dir}/sample_metrics.json" \
      --composition-summary "${NOTES_DIR}/h1a4_${epoch_label}_composition.json" \
      --composition-key refined_pt \
      --crysllmgen-metrics "${NOTES_DIR}/h1a4_${epoch_label}_crysllmgen_metrics.json" \
      --min-comp-valid 0.85 \
      --min-crys-comp-valid 85.0 \
      --min-crys-struct-valid 94.0 \
      --min-crys-cov-recall 85.0 \
      --output-json "${NOTES_DIR}/h1a4_${epoch_label}_refined_gate.json" || true

  run_a100_for_refined "${epoch_label}" "${refined_pt}"
done

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" EPOCHS="${EPOCHS}" python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])
summary = {"run_id": os.environ["RUN_ID"], "epochs": {}}
for epoch in range(1, int(os.environ["EPOCHS"]) + 1):
    label = f"epoch{epoch}"
    item = {}
    for name in ("planner_gate", "hybrid_gate", "refined_gate", "crysllmgen_metrics", "composition", "a100_sun_summary"):
        path = notes / f"h1a4_{label}_{name}.json"
        if path.exists():
            item[name] = json.loads(path.read_text(encoding="utf-8"))
    pt = notes / f"h1a4_{label}_refined_pt.txt"
    if pt.exists():
        item["refined_pt"] = pt.read_text(encoding="utf-8").strip()
    summary["epochs"][label] = item
notes.joinpath("h1a4_joint_basin_planner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# H1-A4 Joint-Basin Planner Report", "", f"- RUN_ID: `{os.environ['RUN_ID']}`", ""]
for label, item in summary["epochs"].items():
    crys = ((item.get("crysllmgen_metrics") or {}).get("metrics") or {})
    a100 = item.get("a100_sun_summary") or {}
    strict = (a100.get("dlm_strict") or {}).get("coverage-adjusted_sun_estimate_pct")
    meta = (a100.get("dlm_meta_like") or {}).get("coverage-adjusted_sun_estimate_pct")
    lines.extend([
        f"## {label}",
        "",
        f"- comp_valid: `{crys.get('comp_valid')}`",
        f"- struct_valid: `{crys.get('struct_valid')}`",
        f"- cov_recall: `{crys.get('cov_recall')}`",
        f"- density WDist: `{crys.get('wdist_density')}`",
        f"- A100 strict adjusted: `{strict}`",
        f"- A100 meta-like adjusted: `{meta}`",
        "",
    ])
notes.joinpath("h1a4_joint_basin_planner_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

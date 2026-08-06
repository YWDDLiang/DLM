#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_WARM_START="${DLM_WARM_START:-none}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
PLANNER_DATA_DIR="${PLANNER_DATA_DIR:-data/dlm_sft/mp_20_h2p1_rich_planner_reference}"
H2_DATA_DIR="${H2_DATA_DIR:-data/dlm_sft/mp_20_h2p1_plaintext_dlm}"
DEFAULT_H1A4_CHECKPOINT="${DEFAULT_H1A4_CHECKPOINT:-runs/20260604_h1a4_joint_basin_planner/outputs/h1a4_epoch1_llama_rich_sft/final}"
DEFAULT_H1A2_CHECKPOINT="${DEFAULT_H1A2_CHECKPOINT:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final}"
PLANNER_CHECKPOINT_PATH="${PLANNER_CHECKPOINT_PATH:-}"
if [ -z "${PLANNER_CHECKPOINT_PATH}" ]; then
  if [ -d "${DEFAULT_H1A4_CHECKPOINT}" ]; then
    PLANNER_CHECKPOINT_PATH="${DEFAULT_H1A4_CHECKPOINT}"
  else
    PLANNER_CHECKPOINT_PATH="${DEFAULT_H1A2_CHECKPOINT}"
  fi
fi

GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1200}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-1000}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-96}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.9}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
H2_SFT_BATCH_SIZE="${H2_SFT_BATCH_SIZE:-1}"
H2_SFT_GRAD_ACCUM="${H2_SFT_GRAD_ACCUM:-8}"
H2_SFT_LR="${H2_SFT_LR:-1e-5}"
H2_SFT_EPOCHS="${H2_SFT_EPOCHS:-1}"
H2_SFT_MAX_LENGTH="${H2_SFT_MAX_LENGTH:-768}"
H2_GEN_LENGTH="${H2_GEN_LENGTH:-360}"
H2_BLOCK_LENGTH="${H2_BLOCK_LENGTH:-4}"
H2_TEMPERATURE="${H2_TEMPERATURE:-0.7}"
H2_SKIP_GRAPH_VALIDATION="${H2_SKIP_GRAPH_VALIDATION:-0}"
DIFF_STEPS="${DIFF_STEPS:-800}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
RUN_A100_SUN="${RUN_A100_SUN:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${PLANNER_NPROC}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
  echo "GPU counts must be <=2 for this project." >&2
  exit 2
fi
if [ "${H2_GEN_LENGTH}" -gt 512 ]; then
  echo "H2_GEN_LENGTH=${H2_GEN_LENGTH} is unusually high for plain-text proposal; refusing accidental runaway." >&2
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

BASE_MASTER_PORT="${MASTER_PORT:-$((33000 + (${SLURM_JOB_ID:-0} % 7000)))}"
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
    /tmp/h2p1_mp_key_*|/public/home/jiaosz/.cache/codex/h2p1_mp_key_*)
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
  "stage": "h2p1_plaintext_dlm_proposal",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "planner_checkpoint_path": "${PLANNER_CHECKPOINT_PATH}",
  "dlm_model_path": "${DLM_MODEL_PATH}",
  "dlm_warm_start": "${DLM_WARM_START}",
  "h2_data_dir": "${H2_DATA_DIR}",
  "sample_count": int("${SAMPLE_COUNT}"),
  "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
  "representation": {
    "dlm_output": "CrysLLMGen plain text",
    "dense_geometry_special_tokens": False,
    "gold_plan_at_sampling": False,
    "sampling_filter_or_topk": False
  },
  "a100_baseline_reference_not_rerun": {
    "strict_adjusted_pct": 9.31,
    "meta_like_adjusted_pct": 47.67,
    "r5c_conditional_strict_adjusted_pct": 10.61,
    "r5c_conditional_meta_like_adjusted_pct": 74.38
  }
}
Path("${NOTES_DIR}/h2p1_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

activate_conda_env "${MAIN_ENV_NAME}"

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
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h2_plaintext_dlm tests.test_h1_llm_planner tests.test_crysllmgen_text'

test -d "${PLANNER_CHECKPOINT_PATH}"

if [ ! -f "${PLANNER_DATA_DIR}/_SUCCESS" ]; then
  run_logged "${LOG_DIR}/build_h1_rich_planner_data_for_h2p1.log" \
    python scripts/build_h1_llm_formula_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${PLANNER_DATA_DIR}" \
      --tokenizer-path "${PLANNER_MODEL_PATH}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id \
      --mixture direct_plan,correct_plan,consistency_explain,formula_count_check \
      --direct-plan-weight 1.0 \
      --correct-plan-weight 0.5 \
      --consistency-explain-weight 0.25 \
      --formula-count-check-weight 0.25
fi

if [ ! -f "${H2_DATA_DIR}/_SUCCESS" ]; then
  build_args=(python scripts/build_h2_plaintext_dlm_sft_data.py
    --input-dir "${INPUT_CSV_DIR}"
    --output-dir "${H2_DATA_DIR}"
    --tokenizer-path "${DLM_MODEL_PATH}")
  if [ "${H2_SKIP_GRAPH_VALIDATION}" = "1" ]; then
    build_args+=(--skip-graph-validation)
  fi
  run_logged "${LOG_DIR}/build_h2p1_plaintext_dlm_data.log" "${build_args[@]}"
fi

TRAIN_DIR="${OUT_DIR}/h2p1_plaintext_dlm_sft"
if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -d "${TRAIN_DIR}/final" ]; then
  sft_cmd=(torchrun --nproc_per_node="${DLM_NPROC}" scripts/llada_sft.py
    --model-path "${DLM_MODEL_PATH}"
    --data-dir "${H2_DATA_DIR}"
    --output-dir "${TRAIN_DIR}"
    --representation crysllmgen_text
    --max-length "${H2_SFT_MAX_LENGTH}"
    --epochs "${H2_SFT_EPOCHS}"
    --batch-size "${H2_SFT_BATCH_SIZE}"
    --grad-accum "${H2_SFT_GRAD_ACCUM}"
    --lr "${H2_SFT_LR}"
    --save-embedding-layers false)
  if [ -n "${DLM_WARM_START}" ] && [ "${DLM_WARM_START}" != "none" ]; then
    sft_cmd+=(--checkpoint-path "${DLM_WARM_START}")
  fi
  next_port
  run_logged "${LOG_DIR}/h2p1_plaintext_dlm_sft.log" env MASTER_PORT="${NEXT_PORT}" "${sft_cmd[@]}"
fi
test -d "${TRAIN_DIR}/final"

PLANNER_DIR="${OUT_DIR}/h2p1_planner${SAMPLE_COUNT}"
if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${PLANNER_DIR}/sample_metrics.json" ]; then
  next_port
  run_logged "${LOG_DIR}/h2p1_planner_sample${SAMPLE_COUNT}.log" \
    torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
      --model-path "${PLANNER_MODEL_PATH}" \
      --checkpoint-path "${PLANNER_CHECKPOINT_PATH}" \
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
fi

run_logged "${LOG_DIR}/h2p1_planner_gate.log" \
  python scripts/evaluate_h1_planner_gate.py \
    --sample-metrics "${PLANNER_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${PLANNER_DIR}/raw_generations.jsonl" \
    --teacher-jsonl "${PLANNER_DATA_DIR}/train.jsonl" \
    --output-json "${NOTES_DIR}/h2p1_planner_gate.json" \
    --output-md "${NOTES_DIR}/h2p1_planner_gate.md" || true

H2_SAMPLE_DIR="${OUT_DIR}/h2p1_plaintext_dlm_sample${SAMPLE_COUNT}"
if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${H2_SAMPLE_DIR}/sample_metrics.json" ]; then
  next_port
  run_logged "${LOG_DIR}/h2p1_plaintext_dlm_sample${SAMPLE_COUNT}.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_h2_plaintext_dlm.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${TRAIN_DIR}/final" \
      --prompt-jsonl "${PLANNER_DIR}/plans_for_dlm.jsonl" \
      --output-dir "${H2_SAMPLE_DIR}" \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${DLM_BATCH_SIZE:-8}" \
      --gen-length "${H2_GEN_LENGTH}" \
      --block-length "${H2_BLOCK_LENGTH}" \
      --temperature "${H2_TEMPERATURE}"
fi

run_logged "${LOG_DIR}/h2p1_plaintext_gate.log" \
  python scripts/evaluate_h2_plaintext_gate.py \
    --planner-gate-json "${NOTES_DIR}/h2p1_planner_gate.json" \
    --h2-sample-metrics "${H2_SAMPLE_DIR}/sample_metrics.json" \
    --output-json "${NOTES_DIR}/h2p1_plaintext_gate.json" \
    --output-md "${NOTES_DIR}/h2p1_plaintext_gate.md" || true

REFINED_DIR="${OUT_DIR}/h2p1_refined${REFINE_MAX_PROPOSALS}"
if [ "${SKIP_COMPLETED}" != "1" ] || ! find "${REFINED_DIR}" -maxdepth 1 -type f -name 'dlm_refined_mp_*.pt' ! -name '*.rank*.pt' | grep -q .; then
  next_port
  run_logged "${LOG_DIR}/h2p1_refine${REFINE_MAX_PROPOSALS}.log" \
    torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
      --proposal-graphs "${H2_SAMPLE_DIR}/proposal_graphs.pt" \
      --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
      --output-dir "${REFINED_DIR}" \
      --max-proposals "${REFINE_MAX_PROPOSALS}" \
      --diff-steps "${DIFF_STEPS}"
fi

REFINED_PT="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
echo "${REFINED_PT}" > "${NOTES_DIR}/h2p1_refined_pt.txt"

run_logged "${LOG_DIR}/h2p1_crysllmgen_metrics.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/h2p1_crysllmgen_metrics.json"

run_logged "${LOG_DIR}/h2p1_composition.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${H2_SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${H2_SAMPLE_DIR}/raw_generations.jsonl" \
    --text-key text \
    --refined-pt "${REFINED_PT}" \
    --representation crysllmgen_text \
    --refined-world-size "${REFINE_NPROC}" \
    --output-json "${NOTES_DIR}/h2p1_composition.json" \
    --output-md "${NOTES_DIR}/h2p1_composition.md"

run_logged "${LOG_DIR}/h2p1_refined_gate.log" \
  python scripts/evaluate_r5b_gate.py \
    --mode refined1000 \
    --sample-metrics "${H2_SAMPLE_DIR}/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/h2p1_composition.json" \
    --composition-key refined_pt \
    --crysllmgen-metrics "${NOTES_DIR}/h2p1_crysllmgen_metrics.json" \
    --min-comp-valid 0.85 \
    --min-crys-comp-valid 85.0 \
    --min-crys-struct-valid 94.0 \
    --min-crys-cov-recall 85.0 \
    --output-json "${NOTES_DIR}/h2p1_refined_gate.json" || true

run_a100_for_refined() {
  if [ "${RUN_A100_SUN}" != "1" ]; then
    return 0
  fi
  activate_conda_env "${A100_ENV_NAME}"
  local a100_run_id="${RUN_ID}-a100"
  local a100_notes="runs/${a100_run_id}/notes"
  mkdir -p "${a100_notes}"
  run_logged "${LOG_DIR}/h2p1_a100_cache_missing_pre.log" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
      --eval-dir reference/a100_eval_sun \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --run "dlm=${REFINED_PT}" \
      --summary-json "${NOTES_DIR}/h2p1_a100_cache_missing_pre.json"
  local missing_count
  missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/h2p1_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
  if [ "${missing_count}" != "0" ]; then
    if [ -n "${MP_API_KEY_FILE}" ] && [ -s "${MP_API_KEY_FILE}" ]; then
      run_logged "${LOG_DIR}/h2p1_a100_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${REFINED_PT}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/h2p1_a100_cache_enrich.json"
    elif [ "${ALLOW_MISSING_CACHE}" != "1" ]; then
      echo "A100 cache missing ${missing_count} entries for h2p1; provide MP_API_KEY_FILE or set ALLOW_MISSING_CACHE=1." >&2
      exit 2
    fi
  fi
  run_logged "${LOG_DIR}/h2p1_a100_sun.log" \
    env RUN_ID="${a100_run_id}" DLM_PT="${REFINED_PT}" PYTHON_BIN=python \
      MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
      ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
      bash scripts/a800/run_a100_eval_sun_dlm_only.sh
  cp "${a100_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/h2p1_a100_sun_summary.json"
  cp "${a100_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/h2p1_a100_sun_strict_summary.md"
  cp "${a100_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/h2p1_a100_sun_meta_like_summary.md"
  activate_conda_env "${MAIN_ENV_NAME}"
}

run_a100_for_refined

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])
summary = {"run_id": os.environ["RUN_ID"]}
for name in (
    "planner_gate",
    "plaintext_gate",
    "refined_gate",
    "crysllmgen_metrics",
    "composition",
    "a100_sun_summary",
):
    path = notes / f"h2p1_{name}.json"
    if path.exists():
        summary[name] = json.loads(path.read_text(encoding="utf-8"))
pt = notes / "h2p1_refined_pt.txt"
if pt.exists():
    summary["refined_pt"] = pt.read_text(encoding="utf-8").strip()
notes.joinpath("h2p1_plaintext_dlm_proposal_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
crys = ((summary.get("crysllmgen_metrics") or {}).get("metrics") or {})
a100 = summary.get("a100_sun_summary") or {}
strict = (a100.get("dlm_strict") or {}).get("coverage-adjusted_sun_estimate_pct")
meta = (a100.get("dlm_meta_like") or {}).get("coverage-adjusted_sun_estimate_pct")
lines = [
    "# H2-P1 Plain-Text DLM Proposal Report",
    "",
    f"- RUN_ID: `{os.environ['RUN_ID']}`",
    f"- comp_valid: `{crys.get('comp_valid')}`",
    f"- struct_valid: `{crys.get('struct_valid')}`",
    f"- cov_recall: `{crys.get('cov_recall')}`",
    f"- density WDist: `{crys.get('wdist_density')}`",
    f"- A100 strict adjusted: `{strict}`",
    f"- A100 meta-like adjusted: `{meta}`",
    "",
]
notes.joinpath("h2p1_plaintext_dlm_proposal_report.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

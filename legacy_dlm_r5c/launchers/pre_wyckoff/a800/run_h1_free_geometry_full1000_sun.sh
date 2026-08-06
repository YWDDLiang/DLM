#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
H1A2_PLANS_JSONL="${H1A2_PLANS_JSONL:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
LABEL="${LABEL:-ablation_no_freeze_plan_composition}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1200}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-1000}"
GPU_COUNT="${GPU_COUNT:-2}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
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

SAMPLE_DIR="${OUT_DIR}/${LABEL}_sample${SAMPLE_COUNT}"
REFINED_DIR="${OUT_DIR}/${LABEL}_refined${REFINE_MAX_PROPOSALS}"
A100_CHILD_RUN_ID="${RUN_ID}-${LABEL}-a100"

BASE_MASTER_PORT="${MASTER_PORT:-$((30000 + (${SLURM_JOB_ID:-0} % 10000)))}"
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
    /tmp/freegeo_full1000_mp_key_*|/public/home/jiaosz/.cache/codex/freegeo_full1000_mp_key_*)
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
    "stage": "h1_free_geometry_full1000_a100_sun",
    "label": "${LABEL}",
    "sample_count": int("${SAMPLE_COUNT}"),
    "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
    "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
    "h1a2_plans_jsonl": "${H1A2_PLANS_JSONL}",
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "sampling_constraints": {
        "freeze_plan_composition": False,
        "duplicate_coordinate_mask": True,
        "lattice_volume_mask": True,
        "generation_schedule": "exact-plan",
    },
    "a100": {
        "mp_cache_path": "${MP_CACHE_PATH}",
        "global_relax_cache_path": "${GLOBAL_RELAX_CACHE_PATH}",
        "mp_api_key_file_present": bool("${MP_API_KEY_FILE}"),
        "allow_missing_cache": "${ALLOW_MISSING_CACHE}",
    },
    "baseline_reference_not_rerun": {
        "strict_adjusted_pct": 9.31,
        "meta_like_adjusted_pct": 47.67,
    },
}
Path("${NOTES_DIR}/h1_free_geometry_full1000_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

activate_conda_env "${MAIN_ENV_NAME}"
python -V | tee -a "${LOG_DIR}/python_version_main.log"

test -f "${H1A2_PLANS_JSONL}"
test -d "${DLM_CHECKPOINT_PATH}"
test -f "${CRYSLLMGEN_CHECKPOINT}"

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/sample_llada_r5_exact_length.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py \
    reference/a100_eval_sun/eval_sun.py \
    reference/a100_eval_sun/eval_sun_resumable.py

PLAN_LINES="$(wc -l < "${H1A2_PLANS_JSONL}")"
echo "plan_lines=${PLAN_LINES}" | tee "${NOTES_DIR}/plan_lines.txt"

if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${SAMPLE_DIR}/sample_metrics.json" ]; then
  next_port
  run_logged "${LOG_DIR}/${LABEL}_sample${SAMPLE_COUNT}.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_CHECKPOINT_PATH}" \
      --prompt-jsonl "${H1A2_PLANS_JSONL}" \
      --output-dir "${SAMPLE_DIR}" \
      --body-prompt-style full_plan_state \
      --num-samples "${SAMPLE_COUNT}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --temperature "${DLM_TEMPERATURE}" \
      --no-freeze-plan-composition \
      --duplicate-coordinate-mask \
      --lattice-volume-mask \
      --generation-schedule exact-plan
else
  echo "Skipping completed sample: ${SAMPLE_DIR}/sample_metrics.json" | tee -a "${LOG_DIR}/${LABEL}_skip.log"
fi

run_logged "${LOG_DIR}/${LABEL}_raw_composition.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --text-key text \
    --representation dynamic_v1 \
    --output-json "${NOTES_DIR}/${LABEL}_composition_raw.json" \
    --output-md "${NOTES_DIR}/${LABEL}_composition_raw.md"

GRAPH_COUNT="$(python - <<PY
import torch
from pathlib import Path
path = Path("${SAMPLE_DIR}/proposal_graphs.pt")
graphs = torch.load(path, map_location="cpu") if path.exists() else []
print(len(graphs))
PY
)"
echo "graph_count=${GRAPH_COUNT}" | tee "${NOTES_DIR}/graph_count.txt"

if [ "${SKIP_COMPLETED}" != "1" ] || ! find "${REFINED_DIR}" -maxdepth 1 -type f -name 'dlm_refined_mp_*.pt' ! -name '*.rank*.pt' | grep -q .; then
  next_port
  run_logged "${LOG_DIR}/${LABEL}_refine${REFINE_MAX_PROPOSALS}.log" \
    torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
      --proposal-graphs "${SAMPLE_DIR}/proposal_graphs.pt" \
      --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
      --output-dir "${REFINED_DIR}" \
      --max-proposals "${REFINE_MAX_PROPOSALS}" \
      --diff-steps "${DIFF_STEPS}"
else
  echo "Skipping completed refine in ${REFINED_DIR}" | tee -a "${LOG_DIR}/${LABEL}_skip.log"
fi

REFINED_PT="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
echo "${REFINED_PT}" > "${NOTES_DIR}/refined_pt.txt"

run_logged "${LOG_DIR}/${LABEL}_crysllmgen_metrics${REFINE_MAX_PROPOSALS}.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics${REFINE_MAX_PROPOSALS}.json"

run_logged "${LOG_DIR}/${LABEL}_composition_refined.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --text-key text \
    --refined-pt "${REFINED_PT}" \
    --representation dynamic_v1 \
    --refined-world-size "${REFINE_NPROC}" \
    --output-json "${NOTES_DIR}/composition${REFINE_MAX_PROPOSALS}.json" \
    --output-md "${NOTES_DIR}/composition${REFINE_MAX_PROPOSALS}.md"

run_logged "${LOG_DIR}/${LABEL}_refined_gate.log" \
  python scripts/evaluate_r5b_gate.py \
    --mode refined1000 \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/composition${REFINE_MAX_PROPOSALS}.json" \
    --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics${REFINE_MAX_PROPOSALS}.json" \
    --output-json "${NOTES_DIR}/refined${REFINE_MAX_PROPOSALS}_gate.json" || true

activate_conda_env "${A100_ENV_NAME}"
python -V | tee -a "${LOG_DIR}/python_version_a100.log"

run_logged "${LOG_DIR}/a100_cache_missing_pre.log" \
  python scripts/a800/check_a100_eval_sun_cache_missing.py \
    --eval-dir reference/a100_eval_sun \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --cache-path "${MP_CACHE_PATH}" \
    --run "dlm=${REFINED_PT}" \
    --summary-json "${NOTES_DIR}/a100_cache_missing_pre.json"

MISSING_COUNT="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
if [ "${MISSING_COUNT}" != "0" ]; then
  if [ -n "${MP_API_KEY_FILE}" ] && [ -s "${MP_API_KEY_FILE}" ]; then
    run_logged "${LOG_DIR}/a100_cache_enrich.log" \
      python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
        --eval-dir reference/a100_eval_sun \
        --gen-file "${REFINED_PT}" \
        --train-csv reference/crysllmgen/data/mp_20/train.csv \
        --cache-path "${MP_CACHE_PATH}" \
        --key-file "${MP_API_KEY_FILE}" \
        --summary-json "${NOTES_DIR}/a100_cache_enrich.json"
  elif [ "${ALLOW_MISSING_CACHE}" != "1" ]; then
    echo "A100 cache missing ${MISSING_COUNT} entries and no MP_API_KEY_FILE was provided." >&2
    exit 2
  fi
fi

run_logged "${LOG_DIR}/a100_sun.log" \
  env RUN_ID="${A100_CHILD_RUN_ID}" DLM_PT="${REFINED_PT}" PYTHON_BIN=python \
    MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
    ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
    bash scripts/a800/run_a100_eval_sun_dlm_only.sh

cp "runs/${A100_CHILD_RUN_ID}/notes/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/a100_eval_sun_dlm_only_summary.json"
cp "runs/${A100_CHILD_RUN_ID}/notes/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/a100_strict_summary.md"
cp "runs/${A100_CHILD_RUN_ID}/notes/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/a100_meta_like_summary.md"

python - <<PY
import json
from pathlib import Path

notes = Path("${NOTES_DIR}")
summary = {
    "run_id": "${RUN_ID}",
    "label": "${LABEL}",
    "sample_metrics": json.loads(Path("${SAMPLE_DIR}/sample_metrics.json").read_text(encoding="utf-8")),
    "graph_count": int(Path("${NOTES_DIR}/graph_count.txt").read_text().strip().split("=")[-1]),
    "refined_pt": Path("${NOTES_DIR}/refined_pt.txt").read_text().strip(),
    "crysllmgen_metrics": json.loads((notes / "crysllmgen_metrics${REFINE_MAX_PROPOSALS}.json").read_text(encoding="utf-8")),
    "composition": json.loads((notes / "composition${REFINE_MAX_PROPOSALS}.json").read_text(encoding="utf-8")),
    "a100": json.loads((notes / "a100_eval_sun_dlm_only_summary.json").read_text(encoding="utf-8")),
}
(notes / "h1_free_geometry_full1000_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
strict = summary["a100"]["dlm_strict"]
meta = summary["a100"]["dlm_meta_like"]
crys = summary["crysllmgen_metrics"].get("metrics", summary["crysllmgen_metrics"])
lines = [
    "# H1 Free-Geometry Full1000 A100 S.U.N.",
    "",
    f"- RUN_ID: `${summary['run_id']}`",
    f"- label: `${summary['label']}`",
    f"- requested samples: `{summary['sample_metrics'].get('requested_samples')}`",
    f"- graph success: `{summary['sample_metrics'].get('graph_success')}`",
    f"- refined pt: `{summary['refined_pt']}`",
    "",
    "## CrysLLMGen Metrics",
    "",
    f"- comp_valid: `{crys.get('comp_valid')}`",
    f"- struct_valid: `{crys.get('struct_valid')}`",
    f"- cov_recall: `{crys.get('cov_recall')}`",
    f"- cov_precision: `{crys.get('cov_precision')}`",
    f"- wdist_density: `{crys.get('wdist_density')}`",
    "",
    "## A100 S.U.N.",
    "",
    f"- strict adjusted: `{strict.get('coverage-adjusted_sun_estimate_pct')}`",
    f"- strict lower-bound: `{strict.get('full_sun_lower-bound_pct')}`",
    f"- meta-like adjusted: `{meta.get('coverage-adjusted_sun_estimate_pct')}`",
    f"- meta-like lower-bound: `{meta.get('full_sun_lower-bound_pct')}`",
    f"- Novel+Unique: `{strict.get('novel_+_unique_pct')}`",
    f"- hull eval: `{strict.get('e_hull_evaluated')}`",
    "",
]
(notes / "h1_free_geometry_full1000_report.md").write_text("\n".join(lines), encoding="utf-8")
PY

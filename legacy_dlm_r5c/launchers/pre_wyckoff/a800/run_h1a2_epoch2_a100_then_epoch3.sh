#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PREV_RUN_ID="${PREV_RUN_ID:-20260603_034533-h1a2-epoch2-3-fullmetrics}"
EPOCH3_RUN_ID="${EPOCH3_RUN_ID:-${RUN_ID}-epoch3-fullmetrics}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}"

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
  echo "prev_run_id=${PREV_RUN_ID}"
  echo "epoch3_run_id=${EPOCH3_RUN_ID}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort > "${NOTES_DIR}/environment.txt"

PREV_RUN_DIR="runs/${PREV_RUN_ID}"
EPOCH2_REFINED_PT="${PREV_RUN_DIR}/outputs/h1a2_epoch2_refined1000/dlm_refined_mp_1000.pt"
EPOCH2_CHECKPOINT="${PREV_RUN_DIR}/outputs/h1a2_epoch2_llama_rich_sft/final"
test -f "${EPOCH2_REFINED_PT}"
test -d "${EPOCH2_CHECKPOINT}"

activate_conda_env "${A100_ENV_NAME}"

run_logged "${LOG_DIR}/h1a2_epoch2_a100_cache_missing_pre.log" \
  python scripts/a800/check_a100_eval_sun_cache_missing.py \
    --eval-dir reference/a100_eval_sun \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --cache-path "${MP_CACHE_PATH}" \
    --run "dlm=${EPOCH2_REFINED_PT}" \
    --summary-json "${NOTES_DIR}/h1a2_epoch2_a100_cache_missing_pre.json"

missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/h1a2_epoch2_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
if [ "${missing_count}" != "0" ]; then
  if [ -z "${MP_API_KEY_FILE}" ] || [ ! -s "${MP_API_KEY_FILE}" ]; then
    echo "A100 MP cache missing entries for epoch2, but MP_API_KEY_FILE is empty or missing." >&2
    exit 2
  fi
  run_logged "${LOG_DIR}/h1a2_epoch2_a100_cache_enrich.log" \
    python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
      --eval-dir reference/a100_eval_sun \
      --gen-file "${EPOCH2_REFINED_PT}" \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --key-file "${MP_API_KEY_FILE}" \
      --summary-json "${NOTES_DIR}/h1a2_epoch2_a100_cache_enrich.json"
fi

a100_run_id="${RUN_ID}-a100-epoch2"
a100_notes="runs/${a100_run_id}/notes"
mkdir -p "${a100_notes}"
run_logged "${LOG_DIR}/h1a2_epoch2_a100_sun.log" \
  env RUN_ID="${a100_run_id}" DLM_PT="${EPOCH2_REFINED_PT}" PYTHON_BIN=python \
    MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
    bash scripts/a800/run_a100_eval_sun_dlm_only.sh
cp "${a100_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/h1a2_epoch2_a100_sun_summary.json"
cp "${a100_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/h1a2_epoch2_a100_sun_strict_summary.md"
cp "${a100_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/h1a2_epoch2_a100_sun_meta_like_summary.md"

activate_conda_env "${MAIN_ENV_NAME}"

run_logged "${LOG_DIR}/h1a2_epoch3_fullmetrics.log" \
  env RUN_ID="${EPOCH3_RUN_ID}" \
    EPOCH1_CHECKPOINT="${EPOCH2_CHECKPOINT}" \
    START_EPOCH=3 \
    END_EPOCH=3 \
    MP_API_KEY_FILE="${MP_API_KEY_FILE}" \
    bash scripts/a800/run_h1a2_epoch_extension_fullmetrics.sh

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
summary = {
    "run_id": "${RUN_ID}",
    "prev_run_id": "${PREV_RUN_ID}",
    "epoch3_run_id": "${EPOCH3_RUN_ID}",
}
for name in ("h1a2_epoch2_a100_sun_summary",):
    path = notes / f"{name}.json"
    if path.exists():
        summary[name] = json.loads(path.read_text(encoding="utf-8"))
epoch3_summary = Path("runs/${EPOCH3_RUN_ID}/notes/h1a2_epoch_extension_summary.json")
if epoch3_summary.exists():
    summary["epoch3_summary"] = json.loads(epoch3_summary.read_text(encoding="utf-8"))
notes.joinpath("h1a2_epoch2_a100_epoch3_resume_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

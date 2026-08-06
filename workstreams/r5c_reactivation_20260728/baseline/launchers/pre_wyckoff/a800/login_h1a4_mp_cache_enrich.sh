#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260604_h1a4_joint_basin_planner_clean}"
KEY_FILE="${KEY_FILE:-/public/home/jiaosz/.cache/codex/h1a4_mp_key_20260604}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-crysllm}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR="runs/${PARENT_RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}"

cleanup_key() {
  case "${KEY_FILE:-}" in
    /public/home/jiaosz/.cache/codex/h1a4_mp_key_*)
      [ -f "${KEY_FILE}" ] && rm -f "${KEY_FILE}"
      ;;
  esac
}
trap 'status=$?; cleanup_key; echo "${status}" > "${NOTES_DIR}/h1a4_login_cache_enrich_exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/h1a4_login_cache_enrich_end_time.txt"; exit "${status}"' EXIT

date "+%F %T %Z" > "${NOTES_DIR}/h1a4_login_cache_enrich_start_time.txt"
{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "key_file=${KEY_FILE}"
} > "${NOTES_DIR}/h1a4_login_cache_enrich_host.txt"

if [ ! -s "${KEY_FILE}" ]; then
  echo "Missing MP key file: ${KEY_FILE}" >&2
  exit 3
fi

set +u
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
set -u

python -m py_compile scripts/a800/check_a100_eval_sun_cache_missing.py scripts/a800/enrich_a100_eval_sun_mp_cache.py

run_json_step() {
  local summary_json="$1"
  shift
  set +e
  "$@"
  local status=$?
  set -e
  if [ "${status}" -eq 0 ]; then
    return 0
  fi
  if [ "${status}" -eq 139 ] && [ -s "${summary_json}" ]; then
    echo "WARNING: command exited with 139 after writing ${summary_json}; continuing." >&2
    echo "139_after_summary ${summary_json}" >> "${NOTES_DIR}/h1a4_login_cache_enrich_warnings.txt"
    return 0
  fi
  return "${status}"
}

for ep in 1 2; do
  refined_pt="${RUN_DIR}/outputs/h1a4_epoch${ep}_refined1000/dlm_refined_mp_1000.pt"
  if [ ! -f "${refined_pt}" ]; then
    echo "Missing refined pt for epoch${ep}: ${refined_pt}" >&2
    exit 4
  fi
  pre_json="${NOTES_DIR}/h1a4_epoch${ep}_a100_cache_missing_login_pre.json"
  enrich_json="${NOTES_DIR}/h1a4_epoch${ep}_a100_cache_enrich_login.json"
  post_json="${NOTES_DIR}/h1a4_epoch${ep}_a100_cache_missing_login_post.json"

  run_json_step "${pre_json}" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
    --eval-dir reference/a100_eval_sun \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --cache-path "${MP_CACHE_PATH}" \
    --run "dlm=${refined_pt}" \
    --summary-json "${pre_json}"

  run_json_step "${enrich_json}" \
    python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
    --eval-dir reference/a100_eval_sun \
    --gen-file "${refined_pt}" \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --cache-path "${MP_CACHE_PATH}" \
    --key-file "${KEY_FILE}" \
    --summary-json "${enrich_json}"

  run_json_step "${post_json}" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
    --eval-dir reference/a100_eval_sun \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --cache-path "${MP_CACHE_PATH}" \
    --run "dlm=${refined_pt}" \
    --summary-json "${post_json}"
done

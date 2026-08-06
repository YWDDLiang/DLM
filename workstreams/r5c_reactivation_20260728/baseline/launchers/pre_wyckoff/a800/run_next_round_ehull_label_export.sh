#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260604_next_round_ehull_label_export}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
ENV_NAME="${ENV_NAME:-crysllm}"
EVAL_DIR="${EVAL_DIR:-reference/a100_eval_sun}"
TRAIN_CSV="${TRAIN_CSV:-reference/crysllmgen/data/mp_20/train.csv}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
NOTES_DIR="${RUN_DIR}/notes"
OUT_DIR="${RUN_DIR}/outputs/ehull_labels"
mkdir -p "${LOG_DIR}" "${NOTES_DIR}" "${OUT_DIR}"

JOB_NAME="${SLURM_JOB_NAME:-next-ehull-labels}"
JOB_ID="${SLURM_JOB_ID:-manual}"
FULL_LOG="${LOG_DIR}/${JOB_NAME}-${JOB_ID}.full.log"

{
  echo "===== JOB START ====="
  echo "date=$(date '+%F %T %Z')"
  echo "job_id=${JOB_ID}"
  echo "job_name=${JOB_NAME}"
  echo "nodelist=${SLURM_JOB_NODELIST:-manual}"
  echo "project_root=${PROJECT_ROOT}"
  echo "run_dir=${RUN_DIR}"
  echo "pwd=$(pwd)"
  echo "hostname=$(hostname)"
  echo "whoami=$(whoami)"
  echo
  echo "===== NVIDIA-SMI BEFORE ====="
  nvidia-smi || true
  echo
  echo "===== ENVIRONMENT ====="
  set +u
  if [ -f /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
  else
    source ~/.bashrc
  fi
  set -u
  conda activate "${ENV_NAME}"
  python -V
  python -m pip freeze | sed -n '1,220p' || true
  python -m py_compile scripts/a800/export_ehull_labels_from_relax.py

  declare -a RUN_ARGS=()
  add_run() {
    local label="$1"
    local path="$2"
    if [ -f "${path}" ]; then
      RUN_ARGS+=(--run "${label}=${path}")
      echo "ADD_RUN ${label} ${path}"
    else
      echo "MISSING_RUN ${label} ${path}"
    fi
  }

  add_run r5c_conditional runs/20260531_2200-a100-eval-sun-mpapi-cache-final2/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h1a2_epoch2 runs/20260603_h1a2_epoch2_refined1000_a100_sun/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run freegeo_ablation_default runs/20260603_h1_freegeo_ablation_default_full1000_a100_retry/outputs/dlm_a100_eval_sun/relax_results.jsonl

  add_run h1g1_full_rich runs/20260604_h1g1_robust_exact_dlm_resume-a100-h1g1_full_rich/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h1g1_condition_dropout runs/20260604_h1g1_robust_exact_dlm_resume-a100-h1g1_condition_dropout/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h1g1_formula_volume_sg runs/20260604_h1g1_robust_exact_dlm_resume-a100-h1g1_formula_volume_sg/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h1g1_formula_volume_only runs/20260604_h1g1_robust_exact_dlm_resume-a100-h1g1_formula_volume_only/outputs/dlm_a100_eval_sun/relax_results.jsonl

  add_run h1a4_epoch1 runs/20260604_h1a4_joint_basin_planner_clean-a100-epoch1/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h1a4_epoch2 runs/20260604_h1a4_joint_basin_planner_clean-a100-epoch2/outputs/dlm_a100_eval_sun/relax_results.jsonl
  add_run h2p1_plaintext runs/20260604_h2p1_plaintext_dlm_proposal_clean-a100/outputs/dlm_a100_eval_sun/relax_results.jsonl

  printf '%s\n' "${RUN_ARGS[@]}" > "${NOTES_DIR}/run_args.txt"
  python - <<PY
import json, os, platform
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "stage": "next_round_ehull_label_export_from_relax_results",
    "eval_dir": "${EVAL_DIR}",
    "train_csv": "${TRAIN_CSV}",
    "mp_cache_path": "${MP_CACHE_PATH}",
    "out_dir": "${OUT_DIR}",
    "relax_recomputed": False,
    "host": platform.node(),
    "user": os.environ.get("USER"),
    "cwd": os.getcwd(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
Path("${NOTES_DIR}", "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY

  if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
    echo "No relax_results.jsonl files found." >&2
    exit 2
  fi

  echo
  echo "===== COMMAND ====="
  printf 'python scripts/a800/export_ehull_labels_from_relax.py --eval-dir %q --train-csv %q --mp-cache-path %q --out-dir %q ' \
    "${EVAL_DIR}" "${TRAIN_CSV}" "${MP_CACHE_PATH}" "${OUT_DIR}"
  printf '%q ' "${RUN_ARGS[@]}"
  echo
  set +e
  python scripts/a800/export_ehull_labels_from_relax.py \
    --eval-dir "${EVAL_DIR}" \
    --train-csv "${TRAIN_CSV}" \
    --mp-cache-path "${MP_CACHE_PATH}" \
    --out-dir "${OUT_DIR}" \
    "${RUN_ARGS[@]}"
  command_status=$?
  set -e
  echo "command_status=${command_status}"
  echo
  echo "===== NVIDIA-SMI AFTER ====="
  nvidia-smi || true
  echo "date=$(date '+%F %T %Z')"
  echo "===== JOB END ====="
  exit "${command_status}"
} 2>&1 | tee -a "${FULL_LOG}"

status=${PIPESTATUS[0]}
echo "final_status=${status}" | tee -a "${FULL_LOG}"
exit "${status}"

#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260604_h1a4_joint_basin_planner_clean}"
CANCEL_JOB_ID="${CANCEL_JOB_ID:-}"
JOB_NAME="${JOB_NAME:-h1a4-a100-st}"
SLURM_TIME="${SLURM_TIME:-08:00:00}"
SLURM_MEM="${SLURM_MEM:-160G}"
GPU_COUNT="${GPU_COUNT:-2}"

cd "${PROJECT_ROOT}"

if [ -n "${CANCEL_JOB_ID}" ]; then
  scancel "${CANCEL_JOB_ID}" 2>/dev/null || true
  sleep 2
fi

python -m py_compile \
  scripts/a800/check_a100_eval_sun_cache_missing.py \
  scripts/a800/enrich_a100_eval_sun_mp_cache.py \
  scripts/a800/export_ehull_labels_from_relax.py
bash -n scripts/a800/run_a100_eval_sun_dlm_only.sh scripts/a800/run_h1a4_joint_basin_planner.sh

stamp="$(date +%Y%m%d_%H%M%S)"
for ep in 1 2; do
  out_dir="runs/${PARENT_RUN_ID}-a100-epoch${ep}/outputs/dlm_a100_eval_sun"
  if [ -d "${out_dir}" ]; then
    backup="${out_dir}_stale_before_singlethread_${stamp}"
    mv "${out_dir}" "${backup}"
    echo "MOVED epoch${ep}: ${out_dir} -> ${backup}"
  else
    echo "NO_CURRENT epoch${ep}: ${out_dir}"
  fi
done

GPU_COUNT="${GPU_COUNT}" \
SLURM_TIME="${SLURM_TIME}" \
SLURM_MEM="${SLURM_MEM}" \
JOB_NAME="${JOB_NAME}" \
bash scripts/a800/slurm_submit.sh "${PARENT_RUN_ID}" \
  env SKIP_COMPLETED=1 EPOCHS=2 RUN_A100_SUN=1 ALLOW_MISSING_CACHE=0 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    bash scripts/a800/run_h1a4_joint_basin_planner.sh

squeue -u jiaosz -o '%.18i %.30j %.10T %.10M %R'

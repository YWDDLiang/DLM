#!/usr/bin/env bash
#SBATCH --job-name=projected-force-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=projected-force-preflight-%j.out
#SBATCH --error=projected-force-preflight-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly DATA="${ROOT}/data/force_score_preflight_64x8_v1_20260902/preflight_rows.jsonl"
readonly RUN="${ROOT}/runs/projected_force_score_teacher_preflight_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
trap 'set +e; touch "${RUN}/_FAILED"; exit 1' ERR
[[ "$(wc -l < "${DATA}")" -eq 512 ]]
env PYTHONPATH="${CODE}/src" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${CODE}/scripts/run_projected_force_score_preflight.py" \
  --data-jsonl "${DATA}" --output-dir "${RUN}/result" --device cuda
[[ -f "${RUN}/result/_SUCCESS" ]]
trap - ERR
touch "${RUN}/_SUCCESS"

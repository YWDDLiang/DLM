#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-pfreeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=spad-basin-pfreeze-%j.out
#SBATCH --error=spad-basin-pfreeze-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly PLANS="${ROOT}/runs/full_mp20_reference_bodies_39658/programs/plans_for_dlm.jsonl"
readonly OUTPUT="${ROOT}/cohorts/spad_basin_preflight_train128_v1_20260904"
readonly RUN="${ROOT}/runs/spad_basin_preflight_freeze_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ ! -e "${OUTPUT}" ]]
[[ "$(wc -l <"${PLANS}")" -eq 27136 ]]
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/freeze_spad_basin_preflight_cohort.py" \
  --plans-jsonl "${PLANS}" --output-dir "${OUTPUT}" --seed 20260904 \
  >"${RUN}/logs/freeze.out" 2>"${RUN}/logs/freeze.err"
[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/plans_for_dlm.jsonl")" -eq 128 ]]
printf '%s\n' "${OUTPUT}" >"${RUN}/COHORT_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"

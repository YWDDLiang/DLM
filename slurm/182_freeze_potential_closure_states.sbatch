#!/usr/bin/env bash
#SBATCH --job-name=potential-state-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=potential-state-freeze-%j.out
#SBATCH --error=potential-state-freeze-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_spade_b14c457}"
readonly PREDICTOR="${ROOT}/runs/spad_energy_teacher_39556/predictor_body/raw_generations.jsonl"
readonly OUTPUT="${ROOT}/data/potential_closure_states_v1_20260904"
readonly RUN="${ROOT}/runs/freeze_potential_closure_states_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]]
[[ -f "${SOURCE}/scripts/freeze_potential_closure_states.py" ]]
[[ "$(wc -l <"${PREDICTOR}")" -eq 2048 ]]

cat >"${RUN}/contract.tsv" <<EOF
field	value
source	job39556 train-only Llama programs and native BS bodies
selection	outcome-blind matched 512-source hash order
groups	2048 = 512 x four strata
official_or_test_outcomes	false
EOF

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/freeze_potential_closure_states.py" \
  --predictor-body "${PREDICTOR}" --output-dir "${OUTPUT}" \
  --requested-sources 512 --selection-seed 20260904 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/states.jsonl")" -eq 2048 ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

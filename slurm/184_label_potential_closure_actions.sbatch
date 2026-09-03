#!/usr/bin/env bash
#SBATCH --job-name=potential-raw-labels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=potential-raw-labels-%j.out
#SBATCH --error=potential-raw-labels-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_spade_b14c457}"
readonly ACTIONS="${POTENTIAL_CLOSURE_ACTIONS:?export POTENTIAL_CLOSURE_ACTIONS}"
readonly CANDIDATE_GROUPS="${ACTIONS}/candidate_groups.jsonl"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/label_potential_closure_actions_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/labels"

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

[[ -f "${SOURCE}/scripts/label_potential_closure_actions.py" ]]
[[ -f "${ACTIONS}/_SUCCESS" ]]
[[ "$(wc -l <"${CANDIDATE_GROUPS}")" -eq 2048 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/contract.tsv" <<EOF
field	value
candidate_groups	${CANDIDATE_GROUPS}
label	raw CHGNet E/F/stress only
batch_size	16
device	job-local cuda:0
model494_or_official	false
EOF

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/label_potential_closure_actions.py" \
  --candidate-groups "${CANDIDATE_GROUPS}" --output-dir "${OUTPUT}" \
  --device cuda:0 --batch-size 16 \
  >"${RUN}/label.out" 2>"${RUN}/label.err"

[[ -f "${OUTPUT}/manifest.json" ]]
[[ -f "${OUTPUT}/_SUCCESS" || -f "${OUTPUT}/_BLOCKED" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-list-safety
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=h1a2-list-safety-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${ROOT}/data/d3po_listwise_groups_v1_20260830"
readonly OUTPUT="${ROOT}/data/listwise_safety_v2_20260830"
readonly RUN="${ROOT}/runs/listwise_safety_v2_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly MANIFEST_SHA=4a7b0f8b56e03e5d8dfdf0f515e7b37a7035efea5a6072c03635d8d1cad51fb9
readonly TRAIN_SHA=5d8d4db03725a8439b1498539d6efcfdc5b41b9eac2f186172ca1b48e1f56b8b
readonly VALIDATION_SHA=e20c0ed78b4c8a61bf09de42e4bd5561cdca1f607e992bab5ab91e99160318fc

mkdir -p "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.preparing" && ! -e "${OUTPUT}.FAILED" ]] || exit 2
[[ "$(sha256sum "${SOURCE}/D3PO_LISTWISE_GROUP_MANIFEST.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${SOURCE}/train.jsonl" | awk '{print $1}')" == "${TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${SOURCE}/validation.jsonl" | awk '{print $1}')" == "${VALIDATION_SHA}" ]] || exit 2

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
method	listwise safety v2 raw-CIF augmentation
source	${SOURCE}
source_manifest_sha256	${MANIFEST_SHA}
source_train_sha256	${TRAIN_SHA}
source_validation_sha256	${VALIDATION_SHA}
raw_minimum_distance_angstrom	0.5
raw_minimum_volume_angstrom3	0.1
raw_invalid_is_worst	true
raw_energy	unavailable_explicit_missing
workers	48
gpu_jobs_used	0
holdout_or_test_outcome	unsupported
output	${OUTPUT}
EOF
sha256sum "${SOURCE}/D3PO_LISTWISE_GROUP_MANIFEST.json" \
  "${SOURCE}/train.jsonl" "${SOURCE}/validation.jsonl" \
  "${ROOT}/scripts/build_listwise_safety_v2.py" >"${RUN}/inputs.sha256"

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 "${PYTHON}" \
  "${ROOT}/scripts/build_listwise_safety_v2.py" \
  --listwise-dir "${SOURCE}" --output-dir "${OUTPUT}" --workers 48 \
  >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

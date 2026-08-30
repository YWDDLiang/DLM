#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-canary-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=c3fd-native-canary-freeze-%j.out
#SBATCH --error=c3fd-native-canary-freeze-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly TEACHER="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly PREDICTED="${ROOT}/data/c3fd_native_soft_predictions_v1_20260830"
readonly OUTPUT="${ROOT}/cohorts/c3fd_native_sft_canary_train128_val128_v1_20260831"
readonly RUN="${ROOT}/runs/freeze_c3fd_native_sft_canary_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TEACHER_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly PREDICTED_MANIFEST_SHA=55a7141e8e864390a5af3702f1fc3f68d746e02838c733ff5a842cdcb9d0b84a

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${OUTPUT}" && ! -f "${OUTPUT}/_SUCCESS" ]]; then
    mv "${OUTPUT}" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ -f "${TEACHER}/_SUCCESS" && -f "${PREDICTED}/_SUCCESS" ]] || exit 2
[[ "$(sha256sum "${TEACHER}/manifest.json" | awk '{print $1}')" == "${TEACHER_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${PREDICTED}/manifest.json" | awk '{print $1}')" == "${PREDICTED_MANIFEST_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_sft_canary_freeze_v1\n'
  printf 'teacher_manifest=%s\n' "${TEACHER_MANIFEST_SHA}"
  printf 'predicted_manifest=%s\n' "${PREDICTED_MANIFEST_SHA}"
  printf 'splits=train128,val128\n'
  printf 'freeze_seed=20260831\n'
  printf 'planner_checkpoints=seed17,seed18\n'
  printf 'checkpoint_selection=none\n'
  printf 'policy_or_test_outcomes_read=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir -p "${ROOT}/runs/contracts"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_sft_canary_freeze_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'requested\t256\n'
  printf 'train\t128\n'
  printf 'validation\t128\n'
  printf 'planner_checkpoints\tseed17,seed18\n'
  printf 'outcomes_read\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${TEACHER}/train.jsonl" "${TEACHER}/val.jsonl" \
  "${TEACHER}/manifest.json" "${PREDICTED}/train.jsonl" \
  "${PREDICTED}/val.jsonl" "${PREDICTED}/manifest.json" \
  "${ROOT}/scripts/freeze_c3fd_native_sft_canary.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${ROOT}/scripts/freeze_c3fd_native_sft_canary.py" \
  --teacher-dir "${TEACHER}" --predicted-dir "${PREDICTED}" \
  --output-dir "${OUTPUT}" --per-split 128 --freeze-seed 20260831 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["requested"] == 256; assert x["split_counts"] == {"train":128,"val":128}; assert x["unique_exact_compositions"] == 256; assert x["planner_checkpoints"] == ["seed17","seed18"]; assert x["checkpoint_selection"] == "none"; assert x["policy_or_test_outcomes_read"] is False; assert all(x["gates"].values())' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

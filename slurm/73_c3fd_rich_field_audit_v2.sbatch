#!/usr/bin/env bash
#SBATCH --job-name=h1a2-rich-field-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=h1a2-rich-field-v2-%j.out
#SBATCH --error=h1a2-rich-field-v2-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly CHECKPOINT17="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly CHECKPOINT18="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed18/checkpoint.pt"
readonly OUTPUT="${ROOT}/data/c3fd_rich_field_predictability_v2_20260830"
readonly PREPARING="${OUTPUT}.preparing"
readonly RUN="${ROOT}/runs/c3fd_rich_field_audit_v2_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CONTRACT_SHA=85b237335255e4b29fabb65f47567105a870ef2e676879b825ecbf1b7b9b673b
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly SEED17_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly SEED18_SHA=d2e31996d034ba0f452ba72906539d6067bf3d7bf8969940c0f7961011b02b07
readonly VAL_SHA=67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly SCRIPT_SHA=5675ab0ed42de48fe8587b139222ef2582f282e97d2e2cdf2c037c582aa4941a

mkdir -p "${ROOT}/runs/contracts"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${PREPARING}" ]]; then
    printf '%s\n' "${SLURM_JOB_ID}" >"${PREPARING}/FAILED_JOB_ID"
    mv "${PREPARING}" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

mkdir "${CONTRACT_DIR}"
mkdir "${PREPARING}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\t%s\n' 'h1a2_c3fd_rich_field_predictability_audit_v2'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'seed17_sha256\t%s\n' "${SEED17_SHA}"
  printf 'seed18_sha256\t%s\n' "${SEED18_SHA}"
  printf 'validation_sha256\t%s\n' "${VAL_SHA}"
  printf 'vocabulary_sha256\t%s\n' "${VOCAB_SHA}"
  printf 'script_sha256\t%s\n' "${SCRIPT_SHA}"
  printf 'split\tval\n'
  printf 'batch_size\t256\n'
  printf 'device\tcpu\n'
  printf 'cpus\t16\n'
  printf 'gpu_jobs_used\t0\n'
  printf 'outcomes_read\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ "$(sha256sum "${CHECKPOINT17}" | awk '{print $1}')" == "${SEED17_SHA}" ]] || exit 2
[[ "$(sha256sum "${CHECKPOINT18}" | awk '{print $1}')" == "${SEED18_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/val.jsonl" | awk '{print $1}')" == "${VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')" == "${VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/audit_c3fd_rich_field_predictability.py" | awk '{print $1}')" == "${SCRIPT_SHA}" ]] || exit 2

sha256sum \
  "${CHECKPOINT17}" \
  "${CHECKPOINT18}" \
  "${DATA}/val.jsonl" \
  "${DATA}/vocabulary.json" \
  "${ROOT}/scripts/audit_c3fd_rich_field_predictability.py" \
  >"${RUN}/INPUTS.sha256"

env \
  CUDA_VISIBLE_DEVICES= \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 \
  NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${ROOT}/scripts/audit_c3fd_rich_field_predictability.py" \
  --checkpoint "seed17=${CHECKPOINT17}" \
  --checkpoint "seed18=${CHECKPOINT18}" \
  --data-dir "${DATA}" \
  --split val \
  --batch-size 256 \
  --output-dir "${OUTPUT}" \
  >"${RUN}/audit.out" 2>"${RUN}/audit.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
rmdir "${PREPARING}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

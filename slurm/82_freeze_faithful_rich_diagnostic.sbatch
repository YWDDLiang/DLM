#!/usr/bin/env bash
#SBATCH --job-name=h1a2-faithful-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=h1a2-faithful-data-%j.out
#SBATCH --error=h1a2-faithful-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly R0_SOURCE="${ROOT}/cohorts/rich_recovery_dev_seed19_256_v2_20260830/R0.jsonl"
readonly H0_SOURCE="${REPO}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl"
readonly TOKENIZER=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly OUTPUT="${ROOT}/cohorts/faithful_rich_diagnostic_h0_r0s_256_v2_20260830"
readonly RUN="${ROOT}/runs/freeze_faithful_rich_diagnostic_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R0_SHA=e40a00fa407fff384fbb991f75aece67d49ebb6d41867090a05bc3ca8140355c
readonly H0_SHA=444c909e73f53f47528d7a6a429320cb0dbe68f5974baf0ed54f8c3cb514e7dc
readonly BUILDER_SHA=80bff3506568b01549569f7f6479678b444fb91d01952664111118549ae983cf
readonly RENDERER_SHA=76cc9c04cbd966d3a839f16748326b586f871133d8c8096f1034bc73171e7b37

mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${OUTPUT}.preparing" ]]; then
    mv "${OUTPUT}.preparing" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ ! -e "${OUTPUT}.preparing" ]] || exit 2
[[ "$(sha256sum "${R0_SOURCE}" | awk '{print $1}')" == "${R0_SHA}" ]] || exit 2
[[ "$(sha256sum "${H0_SOURCE}" | awk '{print $1}')" == "${H0_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/freeze_faithful_rich_diagnostic_cohort.py" | awk '{print $1}')" == "${BUILDER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/freeze_rich_recovery_cohort.py" | awk '{print $1}')" == "${RENDERER_SHA}" ]] || exit 2

{
  printf 'schema=h1a2_faithful_rich_diagnostic_freeze_v2\n'
  printf 'r0_sha256=%s\n' "${R0_SHA}"
  printf 'h0_sha256=%s\n' "${H0_SHA}"
  printf 'builder_sha256=%s\n' "${BUILDER_SHA}"
  printf 'renderer_sha256=%s\n' "${RENDERER_SHA}"
  printf 'count=256\n'
  printf 'views=H0,R0S\n'
  printf 'selection=first256_all_parsed_h0_and_all256_frozen_r0\n'
  printf 'r0s_allowed_changes=oxidation_candidates,prototype_key\n'
  printf 'outcomes_read=false\n'
  printf 'official_query=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"

{
  printf 'field\tvalue\n'
  printf 'schema\th1a2_faithful_rich_diagnostic_freeze_v2\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'requested_per_view\t256\n'
  printf 'gpu_jobs_used\t0\n'
  printf 'cpus\t8\n'
  printf 'outcomes_read\tfalse\n'
  printf 'confirmatory_claim\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

sha256sum "${R0_SOURCE}" "${H0_SOURCE}" \
  "${ROOT}/scripts/freeze_faithful_rich_diagnostic_cohort.py" \
  "${ROOT}/scripts/freeze_rich_recovery_cohort.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${PYTHON}" "${ROOT}/scripts/freeze_faithful_rich_diagnostic_cohort.py" \
  --r0-source "${R0_SOURCE}" \
  --h0-source "${H0_SOURCE}" \
  --output-dir "${OUTPUT}" \
  --count 256 \
  --tokenizer-path "${TOKENIZER}" \
  >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
[[ "$(wc -l <"${OUTPUT}/R0S.jsonl")" -eq 256 ]] || exit 3
[[ "$(wc -l <"${OUTPUT}/H0.jsonl")" -eq 256 ]] || exit 3
[[ "$(wc -l <"${OUTPUT}/DIAGNOSTIC_LEDGER.jsonl")" -eq 512 ]] || exit 3
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

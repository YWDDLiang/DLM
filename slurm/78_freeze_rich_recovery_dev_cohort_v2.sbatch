#!/usr/bin/env bash
#SBATCH --job-name=h1a2-freeze-rich-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=h1a2-freeze-rich-v2-%j.out
#SBATCH --error=h1a2-freeze-rich-v2-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly SOURCE="${ROOT}/runs/c3fd_corrected_rich_seed19_38336/plans/plans_for_dlm.jsonl"
readonly MPTRAIN="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/train.jsonl"
readonly C3S17="${ROOT}/runs/c3fd_v25_requested1000_36615/c3_seed17/plans_for_dlm.jsonl"
readonly C3S18="${ROOT}/runs/c3fd_v25_requested1000_36615/c3_seed18/plans_for_dlm.jsonl"
readonly D3MAIN="${ROOT}/cohorts/d3po_test_seed17_256_certified_20260830_v2/D3PO_TEST_CERTIFIED_PLANS.jsonl"
readonly D3SEALED="${ROOT}/cohorts/d3po_fallback_holdout_seed17_256_certified_20260830_v2/D3PO_TEST_CERTIFIED_PLANS.jsonl"
readonly OLDH1="${REPO}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl"
readonly OUTPUT="${ROOT}/cohorts/rich_recovery_dev_seed19_256_v2_20260830"
readonly PREPARING="${OUTPUT}.preparing"
readonly RUN="${ROOT}/runs/freeze_rich_recovery_dev_v2_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CONTRACT_SHA=e74437ae385abad07b4e2e1d0a6d743cfa9f6396b347f1b7bcf8747f6eca3c63
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly SOURCE_SHA=2a72e5b28d8aca4a8016e3b4ea5c3d2210e140e052939fa731b56554eabc507f
readonly MPTRAIN_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly C3S17_SHA=eb7afb731d7158181218f08f9e71c423f723668c77fe15b4f0e0050fb135e353
readonly C3S18_SHA=d0ffb3823e8e91ff226488241ce4f559a557b6757f4159842662a3bb73273ad5
readonly D3MAIN_SHA=93e223c3174539b4ac1e573bfe679f6bc42bfab1a8eb6f5fc36cbf5a3ccede40
readonly D3SEALED_SHA=1b7f7111f75ed5b26cb235274f1b8da70393676898732722c3a63a2f61a35ce0
readonly OLDH1_SHA=444c909e73f53f47528d7a6a429320cb0dbe68f5974baf0ed54f8c3cb514e7dc
readonly BUILDER_SHA=1c35e95efef9b61850d8bb0a2a03d41ba9e1a4bb9dd9fdd9d64b9b7420246732

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
  printf 'schema\t%s\n' 'h1a2_rich_recovery_dev_cohort_v2'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'source_sampling_seed\t19\n'
  printf 'selected\t256\n'
  printf 'selection\tfirst_eligible_source_ordinal\n'
  printf 'minimal_prompt\td3po_c3fd_certificate_adapter\n'
  printf 'rcf_shift\t73\n'
  printf 'rcf_tuple\tmetric_lattice,spacegroup,volume_per_atom\n'
  printf 'outcomes_read\tfalse\n'
  printf 'gpu_jobs_used\t0\n'
} >"${RUN}/scientific_contract.tsv"

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ -f "${ROOT}/runs/c3fd_corrected_rich_seed19_38336/_SUCCESS" ]] || exit 2
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] || exit 2
[[ "$(sha256sum "${MPTRAIN}" | awk '{print $1}')" == "${MPTRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${C3S17}" | awk '{print $1}')" == "${C3S17_SHA}" ]] || exit 2
[[ "$(sha256sum "${C3S18}" | awk '{print $1}')" == "${C3S18_SHA}" ]] || exit 2
[[ "$(sha256sum "${D3MAIN}" | awk '{print $1}')" == "${D3MAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${D3SEALED}" | awk '{print $1}')" == "${D3SEALED_SHA}" ]] || exit 2
[[ "$(sha256sum "${OLDH1}" | awk '{print $1}')" == "${OLDH1_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/freeze_rich_recovery_cohort.py" | awk '{print $1}')" == "${BUILDER_SHA}" ]] || exit 2

sha256sum "${SOURCE}" "${MPTRAIN}" "${C3S17}" "${C3S18}" \
  "${D3MAIN}" "${D3SEALED}" "${OLDH1}" \
  "${ROOT}/scripts/freeze_rich_recovery_cohort.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${PYTHON}" "${ROOT}/scripts/freeze_rich_recovery_cohort.py" \
  --source-plans "${SOURCE}" \
  --blocked "mp20_train=${MPTRAIN}" \
  --blocked "c3fd_seed17_seen=${C3S17}" \
  --blocked "c3fd_seed18_seen=${C3S18}" \
  --blocked "d3po_main=${D3MAIN}" \
  --blocked "d3po_sealed=${D3SEALED}" \
  --blocked "historical_h1a2=${OLDH1}" \
  --count 256 \
  --rcf-shift 73 \
  --output-dir "${OUTPUT}" \
  >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
for view in ledger M0 R0 RCF; do
  [[ "$(wc -l <"${OUTPUT}/${view}.jsonl")" -eq 256 ]] || exit 3
done
rmdir "${PREPARING}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
WORK_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation"
RESTORED_BASELINE_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/baseline"
if [ -d "${RESTORED_BASELINE_ROOT}/crystal_dlm" ]; then
  RUNTIME_ROOT="${RESTORED_BASELINE_ROOT}"
else
  RUNTIME_ROOT="${PROJECT_ROOT}"
fi
RUN_ROOT="${PROJECT_ROOT}/runs/20260729_h1a2c_jointchem_v1"
PATCH_SHA256="${H1A2C_JOINTCHEM_PATCH_SHA256:?H1A2C_JOINTCHEM_PATCH_SHA256 is required}"
CLAIM_DIR="${RUN_ROOT}/submission_claim"
RECORD="${RUN_ROOT}/submission_record.txt"
EPOCH2_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
EPOCH2_ADAPTER_SHA256="65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
SOURCE_DIR="${PROJECT_ROOT}/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base"

cd "${PROJECT_ROOT}"
test ! -e "${RUN_ROOT}"
test "$(sha256sum "${WORK_ROOT}/SOURCE_SHA256.txt" | awk '{print $1}')" = "${PATCH_SHA256}"
(cd "${WORK_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
(cd "${RUNTIME_ROOT}" && sha256sum -c "${WORK_ROOT}/RUNTIME_REQUIRED_SHA256.txt")
test "$(sha256sum "${EPOCH2_ADAPTER}/adapter_model.safetensors" | awk '{print $1}')" = "${EPOCH2_ADAPTER_SHA256}"
test "$(sha256sum "${SOURCE_DIR}/train.jsonl" | awk '{print $1}')" = "d431dfec1de8c3240dbc5648867be1b4b676fd85276e805a177b9944f3a1a157"
test "$(sha256sum "${SOURCE_DIR}/val.jsonl" | awk '{print $1}')" = "59327aa789ae5d2bbb66d8a8f0dc882d594bcc14623aa96ce95076ed1b6fc540"
test "$(sha256sum "${SOURCE_DIR}/test.jsonl" | awk '{print $1}')" = "032845826acf1fcb9e7893fc91da05bc0cd9d2363c80ae459d5c446c3c6d5ea8"

for path in \
  "${WORK_ROOT}/slurm/data.sbatch" \
  "${WORK_ROOT}/slurm/arm.sbatch" \
  "${WORK_ROOT}/slurm/assemble.sbatch"; do
  test -s "${path}"
done

mkdir -p "${RUN_ROOT}/logs"
if ! mkdir "${CLAIM_DIR}" 2>/dev/null; then
  echo "submission claim already exists: ${CLAIM_DIR}" >&2
  exit 2
fi
trap 'status=$?; if [ "${status}" -ne 0 ]; then printf "status=failed_before_complete\nexit_status=%s\nexecution_manifest_sha256=%s\n" "${status}" "${PATCH_SHA256}" > "${RECORD}"; fi; exit "${status}"' EXIT

data_job="$(sbatch --parsable --export=ALL,H1A2C_JOINTCHEM_PATCH_SHA256="${PATCH_SHA256}" "${WORK_ROOT}/slurm/data.sbatch")"
case "${data_job}" in *[!0-9]*|'') echo "invalid data job id: ${data_job}" >&2; exit 3;; esac
arm_job="$(sbatch --parsable --dependency=afterok:"${data_job}" --export=ALL,H1A2C_JOINTCHEM_PATCH_SHA256="${PATCH_SHA256}" "${WORK_ROOT}/slurm/arm.sbatch")"
case "${arm_job}" in *[!0-9]*|'') echo "invalid arm job id: ${arm_job}" >&2; exit 3;; esac
assembly_job="$(sbatch --parsable --dependency=afterok:"${arm_job}" --export=ALL,H1A2C_JOINTCHEM_PATCH_SHA256="${PATCH_SHA256}" "${WORK_ROOT}/slurm/assemble.sbatch")"
case "${assembly_job}" in *[!0-9]*|'') echo "invalid assembly job id: ${assembly_job}" >&2; exit 3;; esac

printf "status=complete\nexecution_manifest_sha256=%s\ninitial_adapter_sha256=%s\ndata_job=%s\narm_array_job=%s\nassembly_job=%s\nautomatic_crystal_evaluation_authorized=false\n" \
  "${PATCH_SHA256}" "${EPOCH2_ADAPTER_SHA256}" "${data_job}" "${arm_job}" "${assembly_job}" > "${RECORD}"
printf "DATA_JOB=%s ARM_ARRAY_JOB=%s ASSEMBLY_JOB=%s\n" "${data_job}" "${arm_job}" "${assembly_job}"

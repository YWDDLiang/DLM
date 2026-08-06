#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260805_h1_crplan_fourarm512_route_amendment_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_fourarm512_route_amendment_v1"
EXPECTED_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
EXPECTED_SOURCE_MANIFEST_SHA256="${1:?expected source-manifest SHA argument}"
EXPECTED_LEDGER_SHA256="${2:?expected science-ledger SHA argument}"

test -d "${RUN_ROOT}"
test -d "${SOURCE_ROOT}"
test -d "${RUN_ROOT}/logs"
test -d "${RUN_ROOT}/status"
test -x "${EXECUTION_DIR}/arm.sbatch"
test -x "${EXECUTION_DIR}/assemble.sbatch"
test ! -e "${RUN_ROOT}/arms"
test ! -e "${RUN_ROOT}/terminal"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/submission_record.partial.json"
test ! -e "${RUN_ROOT}/_SUCCESS"
test ! -e "${RUN_ROOT}/_SCIENTIFIC_STOP"
test ! -e "${RUN_ROOT}/_FAILED"
mkdir "${RUN_ROOT}/.submit_lock"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/\\*//g')"
for partition in gpu normal; do
  if ! printf '%s\n' "${partition_snapshot}" \
    | awk -F'|' -v wanted="${partition}" \
        '$1 == wanted {found=1} END {exit found ? 0 : 1}'; then
    printf '%s\n' "${partition_snapshot}" \
      > "${RUN_ROOT}/status/sinfo_failure.txt"
    exit 3
  fi
done
printf '%s\n' "${partition_snapshot}" \
  > "${RUN_ROOT}/status/sinfo_before_submission.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_submission.txt" | cut -d' ' -f1)"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
observed_ledger_sha="$(sha256sum "${EXECUTION_DIR}/SCIENCE_LEDGER.json" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "${observed_ledger_sha}" = "${EXPECTED_LEDGER_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

array_job_id="$(
  sbatch --parsable \
    --array=0-3%2 \
    --export=ALL,EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_LEDGER_SHA256="${EXPECTED_LEDGER_SHA256}",EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256}" \
    "${EXECUTION_DIR}/arm.sbatch"
)"
test -n "${array_job_id}"

ARRAY_JOB_ID="${array_job_id}" \
SOURCE_SHA="${EXPECTED_SOURCE_MANIFEST_SHA256}" \
LEDGER_SHA="${EXPECTED_LEDGER_SHA256}" \
ADAPTER_SHA="${EXPECTED_ADAPTER_SHA256}" \
SINFO_SHA="${SINFO_SHA}" \
"${EXECUTION_DIR}/write_submission_record.py" \
  --stage array_submitted \
  --output "${RUN_ROOT}/submission_record.partial.json"

assembly_job_id="$(
  sbatch --parsable \
    --dependency=afterany:"${array_job_id}" \
    --export=ALL,EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_LEDGER_SHA256="${EXPECTED_LEDGER_SHA256}",ARRAY_JOB_ID="${array_job_id}" \
    "${EXECUTION_DIR}/assemble.sbatch"
)"
test -n "${assembly_job_id}"

ARRAY_JOB_ID="${array_job_id}" \
ASSEMBLY_JOB_ID="${assembly_job_id}" \
SOURCE_SHA="${EXPECTED_SOURCE_MANIFEST_SHA256}" \
LEDGER_SHA="${EXPECTED_LEDGER_SHA256}" \
ADAPTER_SHA="${EXPECTED_ADAPTER_SHA256}" \
SINFO_SHA="${SINFO_SHA}" \
"${EXECUTION_DIR}/write_submission_record.py" \
  --stage dag_submitted \
  --output "${RUN_ROOT}/submission_record.json"

printf 'array_job_id=%s\nassembly_job_id=%s\n' \
  "${array_job_id}" "${assembly_job_id}"

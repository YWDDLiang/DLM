#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260805_h1_crplan_e1_physical_performance_probe_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_e1_physical_performance_probe_v1"
EXPECTED_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
EXPECTED_SOURCE_MANIFEST_SHA256="${1:?expected source-manifest SHA argument}"

test -d "${RUN_ROOT}"
test -d "${SOURCE_ROOT}"
test -d "${RUN_ROOT}/logs"
test -x "${EXECUTION_DIR}/probe.sbatch"
test ! -e "${RUN_ROOT}/probe"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/_SUCCESS"
test ! -e "${RUN_ROOT}/_FAILED"
mkdir "${RUN_ROOT}/.submit_lock"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/\\*//g')"
if ! printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "gpu" {found=1} END {exit found ? 0 : 1}'; then
  printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_failure.txt"
  exit 3
fi
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_submission.txt"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

job_id="$(
  sbatch --parsable \
    --export=ALL,EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256}",EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256}" \
    "${EXECUTION_DIR}/probe.sbatch"
)"
test -n "${job_id}"

JOB_ID="${job_id}" \
SOURCE_SHA="${EXPECTED_SOURCE_MANIFEST_SHA256}" \
ADAPTER_SHA="${EXPECTED_ADAPTER_SHA256}" \
"${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_e1_physical_performance_probe_v1/write_submission_record.py" \
  --output "${RUN_ROOT}/submission_record.json"
printf '%s\n' "${job_id}"

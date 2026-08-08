#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_exact_identity_copy_repair_v6"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"
PARENT_V5_FAILURE_SHA=3f161979bef1de77351ba9178aa59cbbaa794cfcb29e9f9bb7b11884022d9be8

test -d "${SOURCE_ROOT}"
test -x "${LEGACY_PYTHON}"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test ! -e "${RUN_ROOT}/identity_probe_submission_record.json"
test ! -e "${RUN_ROOT}/probe"
test ! -e "${RUN_ROOT}/status/identity_probe_SUCCESS"
mkdir "${RUN_ROOT}/.submit_identity_probe_lock"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "gpu" && $2 == "up" {found=1} END {exit found ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_identity_probe.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' > "${RUN_ROOT}/status/squeue_before_identity_probe.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_identity_probe.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_identity_probe.txt" | cut -d' ' -f1)"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/a800_runtime_preflight.json" | cut -d' ' -f1)"
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"

PROBE_JOB_ID="$(sbatch --parsable \
  --export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON}" \
  "${EXECUTION_DIR}/identity_probe.sbatch")"
printf '%s\n' "${PROBE_JOB_ID}" > "${RUN_ROOT}/status/submitted_identity_probe_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA LEGACY_PYTHON
export IDENTITY_PROBE_JOB_ID="${PROBE_JOB_ID}" PARENT_V5_FAILURE_SHA
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage identity_probe --output "${RUN_ROOT}/identity_probe_submission_record.json"
sha256sum "${RUN_ROOT}/identity_probe_submission_record.json" \
  > "${RUN_ROOT}/identity_probe_submission_record.sha256"
printf 'identity_probe=%s\n' "${PROBE_JOB_ID}"

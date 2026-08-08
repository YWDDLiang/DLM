#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_identity_repair_v4"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected SOURCE_SHA256.txt digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"

test -d "${RUN_ROOT}"
test -d "${SOURCE_ROOT}"
test -d "${RUN_ROOT}/logs"
test -d "${RUN_ROOT}/status"
test -f "${RUN_ROOT}/source_archive.tar.gz"
test -x "${LEGACY_PYTHON}"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test ! -e "${RUN_ROOT}/snapshot_submission_record.json"
test ! -e "${RUN_ROOT}/legacy_snapshot"
test ! -e "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz"
mkdir "${RUN_ROOT}/.submit_snapshot_lock"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
observed_archive_sha="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "${observed_archive_sha}" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"

test -d "${RUN_ROOT}/preflight"
test ! -e "${RUN_ROOT}/preflight/preflight_snapshot_report.json"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/preflight.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${EXECUTION_DIR}/CONFIG.json" \
  --authorization "${EXECUTION_DIR}/AUTHORIZATION.json" \
  --ledger64 "${EXECUTION_DIR}/LEDGER64.json" \
  --ledger256 "${EXECUTION_DIR}/LEDGER256.json" \
  --legacy-python "${LEGACY_PYTHON}" \
  --model-path "${MODEL_PATH}" --p0-adapter-path "${P0_ADAPTER}" \
  --mp20-dir "${MP20_DIR}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/preflight_snapshot_report.json"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/preflight_snapshot_report.json" | cut -d' ' -f1)"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" \
  | awk -F'|' '$1 == "normal" && $2 == "up" {found=1} END {exit found ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_snapshot.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' \
  > "${RUN_ROOT}/status/squeue_before_snapshot.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_snapshot.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_snapshot.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON}"
SNAPSHOT_JOB_ID="$(sbatch --parsable --export="${common_export}" "${EXECUTION_DIR}/snapshot.sbatch")"
printf '%s\n' "${SNAPSHOT_JOB_ID}" > "${RUN_ROOT}/status/submitted_snapshot_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export SNAPSHOT_JOB_ID
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage legacy_snapshot --output "${RUN_ROOT}/snapshot_submission_record.json"
sha256sum "${RUN_ROOT}/snapshot_submission_record.json" \
  > "${RUN_ROOT}/snapshot_submission_record.sha256"
printf 'snapshot=%s\n' "${SNAPSHOT_JOB_ID}"

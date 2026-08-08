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
FROZEN_WITNESS_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact4_witness_input_v1"
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected SOURCE_SHA256.txt digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"
LOCAL_SMACT4_WITNESS_ROOT="${3:?local SMACT4 witness input root}"
EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256="${4:?local witness manifest digest}"

test -d "${RUN_ROOT}"
test -d "${SOURCE_ROOT}"
test -f "${RUN_ROOT}/snapshot_submission_record.json"
test -f "${RUN_ROOT}/snapshot_submission_record.sha256"
sha256sum -c "${RUN_ROOT}/snapshot_submission_record.sha256"
test -f "${RUN_ROOT}/status/snapshot_SUCCESS"
test -f "${RUN_ROOT}/legacy_snapshot/_SUCCESS"
test -f "${RUN_ROOT}/legacy_snapshot/report.json"
test -x "${LEGACY_PYTHON}"
test "${LOCAL_SMACT4_WITNESS_ROOT}" = "${FROZEN_WITNESS_ROOT}"
test -f "${LOCAL_SMACT4_WITNESS_ROOT}/MANIFEST.json"
test -f "${LOCAL_SMACT4_WITNESS_ROOT}/_SUCCESS"
test "$(sha256sum "${LOCAL_SMACT4_WITNESS_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = "${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test ! -e "${RUN_ROOT}/engineering_submission_record.json"
test ! -e "${RUN_ROOT}/data"
test ! -e "${RUN_ROOT}/smoke"
test ! -e "${RUN_ROOT}/training"
test ! -e "${RUN_ROOT}/planner64"
mkdir "${RUN_ROOT}/.submit_engineering_smoke_lock"

SNAPSHOT_JOB_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/status/submitted_snapshot_job_id.txt")"
case "${SNAPSHOT_JOB_ID}" in ''|*[!0-9]*) exit 3 ;; esac
sacct -n -X -j "${SNAPSHOT_JOB_ID}" -o JobIDRaw,State,ExitCode -P \
  > "${RUN_ROOT}/status/sacct_snapshot_before_engineering_smoke.txt"
test "$(awk -F'|' -v wanted="${SNAPSHOT_JOB_ID}" '$1 == wanted {print $2 "|" $3}' "${RUN_ROOT}/status/sacct_snapshot_before_engineering_smoke.txt")" = "COMPLETED|0:0"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
observed_archive_sha="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "${observed_archive_sha}" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"
PRIOR_SNAPSHOT_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/snapshot_submission_record.json" | cut -d' ' -f1)"

test -d "${RUN_ROOT}/preflight"
test ! -e "${RUN_ROOT}/preflight/preflight_engineering_smoke_report.json"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/preflight.py" \
  --source-root "${SOURCE_ROOT}" --config "${EXECUTION_DIR}/CONFIG.json" \
  --authorization "${EXECUTION_DIR}/AUTHORIZATION.json" \
  --ledger64 "${EXECUTION_DIR}/LEDGER64.json" --ledger256 "${EXECUTION_DIR}/LEDGER256.json" \
  --legacy-python "${LEGACY_PYTHON}" \
  --model-path "${MODEL_PATH}" --p0-adapter-path "${P0_ADAPTER}" \
  --mp20-dir "${MP20_DIR}" --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/preflight_engineering_smoke_report.json"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/preflight_engineering_smoke_report.json" | cut -d' ' -f1)"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
for partition in normal gpu; do
  printf '%s\n' "${partition_snapshot}" \
    | awk -F'|' -v wanted="${partition}" '$1 == wanted && $2 == "up" {found=1} END {exit found ? 0 : 1}'
done
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_engineering_smoke.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' > "${RUN_ROOT}/status/squeue_before_engineering_smoke.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_engineering_smoke.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_engineering_smoke.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON},LOCAL_SMACT4_WITNESS_ROOT=${LOCAL_SMACT4_WITNESS_ROOT},EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256=${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}"
DATA_JOB_ID="$(sbatch --parsable --export="${common_export}" "${EXECUTION_DIR}/data.sbatch")"
printf '%s\n' "${DATA_JOB_ID}" > "${RUN_ROOT}/status/submitted_data_job_id.txt"
SMOKE_JOB_ID="$(sbatch --parsable --array=0-1%2 --dependency=afterok:"${DATA_JOB_ID}" --export="${common_export}" "${EXECUTION_DIR}/smoke.sbatch")"
printf '%s\n' "${SMOKE_JOB_ID}" > "${RUN_ROOT}/status/submitted_smoke_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export DATA_JOB_ID SMOKE_JOB_ID PRIOR_SNAPSHOT_SUBMISSION_SHA LOCAL_SMACT4_WITNESS_ROOT
export LOCAL_SMACT4_WITNESS_MANIFEST_SHA="${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}"
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage engineering_smoke --output "${RUN_ROOT}/engineering_submission_record.json"
sha256sum "${RUN_ROOT}/engineering_submission_record.json" > "${RUN_ROOT}/engineering_submission_record.sha256"
printf 'data=%s\nsmoke=%s\n' "${DATA_JOB_ID}" "${SMOKE_JOB_ID}"

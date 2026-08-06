#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260806_h1_nocharge_ion_aux_sft_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_nocharge_ion_aux_sft_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected SOURCE_SHA256.txt digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"
SMACT4_PYTHON="${3:?exact SMACT 4.0.0 Python path}"

test -d "${RUN_ROOT}"
test -d "${SOURCE_ROOT}"
test -d "${RUN_ROOT}/logs"
test -d "${RUN_ROOT}/status"
test -f "${RUN_ROOT}/source_archive.tar.gz"
test -x "${LEGACY_PYTHON}"
test -x "${SMACT4_PYTHON}"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/planner64"
test ! -e "${RUN_ROOT}/data"
test ! -e "${RUN_ROOT}/training"
test ! -e "${RUN_ROOT}/smoke"
mkdir "${RUN_ROOT}/.submit_initial64_lock"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
observed_archive_sha="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "${observed_archive_sha}" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"

mkdir "${RUN_ROOT}/preflight"
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
  --smact4-python "${SMACT4_PYTHON}" \
  --model-path "${MODEL_PATH}" \
  --p0-adapter-path "${P0_ADAPTER}" \
  --mp20-dir "${MP20_DIR}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/preflight_report.json"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
for partition in gpu normal; do
  printf '%s\n' "${partition_snapshot}" \
    | awk -F'|' -v wanted="${partition}" '$1 == wanted {found=1} END {exit found ? 0 : 1}'
done
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_initial64.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_initial64.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON},SMACT4_PYTHON=${SMACT4_PYTHON}"
DATA_JOB_ID="$(sbatch --parsable --export="${common_export}" "${EXECUTION_DIR}/data.sbatch")"
printf '%s\n' "${DATA_JOB_ID}" > "${RUN_ROOT}/status/submitted_data_job_id.txt"
SMOKE_JOB_ID="$(sbatch --parsable --array=0-1%2 --dependency=afterok:"${DATA_JOB_ID}" --export="${common_export}" "${EXECUTION_DIR}/smoke.sbatch")"
printf '%s\n' "${SMOKE_JOB_ID}" > "${RUN_ROOT}/status/submitted_smoke_job_id.txt"
TRAIN_JOB_ID="$(sbatch --parsable --array=0-1%2 --dependency=afterok:"${SMOKE_JOB_ID}" --export="${common_export}" "${EXECUTION_DIR}/train.sbatch")"
printf '%s\n' "${TRAIN_JOB_ID}" > "${RUN_ROOT}/status/submitted_train_job_id.txt"
PLANNER_JOB_ID="$(sbatch --parsable --array=0-2%2 --dependency=afterok:"${TRAIN_JOB_ID}" --export="${common_export},EXPECTED_LEDGER_SHA256=${LEDGER64_SHA}" "${EXECUTION_DIR}/planner64.sbatch")"
printf '%s\n' "${PLANNER_JOB_ID}" > "${RUN_ROOT}/status/submitted_planner64_job_id.txt"
ASSEMBLY_JOB_ID="$(sbatch --parsable --dependency=afterany:"${PLANNER_JOB_ID}" --export="${common_export},EXPECTED_LEDGER_SHA256=${LEDGER64_SHA}" "${EXECUTION_DIR}/assemble64.sbatch")"
printf '%s\n' "${ASSEMBLY_JOB_ID}" > "${RUN_ROOT}/status/submitted_assemble64_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON SMACT4_PYTHON SINFO_SHA
export DATA_JOB_ID SMOKE_JOB_ID TRAIN_JOB_ID PLANNER_JOB_ID ASSEMBLY_JOB_ID
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage initial64 --output "${RUN_ROOT}/submission_record.json"
sha256sum "${RUN_ROOT}/submission_record.json" > "${RUN_ROOT}/submission_record.sha256"
printf 'data=%s\nsmoke=%s\ntrain=%s\nplanner64=%s\nassemble64=%s\n' \
  "${DATA_JOB_ID}" "${SMOKE_JOB_ID}" "${TRAIN_JOB_ID}" "${PLANNER_JOB_ID}" "${ASSEMBLY_JOB_ID}"

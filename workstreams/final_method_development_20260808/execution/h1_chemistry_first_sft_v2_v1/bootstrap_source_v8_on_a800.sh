#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
TRANSFER_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_transfer_input_optimizer_zero_lr_audit_v8"
STAGING_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_source_staging_optimizer_zero_lr_audit_v8"
FREEZE_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_source_freeze_optimizer_zero_lr_audit_v8"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SOURCE_INPUT_ARCHIVE="${1:?source input archive path}"
EXPECTED_SOURCE_INPUT_SHA256="${2:?source input archive SHA256}"
EXPECTED_LEGACY_EVALUATOR_SHA256=ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178
LEGACY_EVALUATOR_ENTRY=crystal_dlm/composition_validity.py

test -x "${LEGACY_PYTHON}"
test -d "${TRANSFER_ROOT}"
test -f "${SOURCE_INPUT_ARCHIVE}"
for path in "${STAGING_ROOT}" "${FREEZE_ROOT}" "${RUN_ROOT}"; do
  test ! -e "${path}"
done
test "$(sha256sum "${SOURCE_INPUT_ARCHIVE}" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INPUT_SHA256}"
test "$(tar -xOf "${SOURCE_INPUT_ARCHIVE}" "${LEGACY_EVALUATOR_ENTRY}" | sha256sum | cut -d' ' -f1)" = \
  "${EXPECTED_LEGACY_EVALUATOR_SHA256}"

mkdir "${STAGING_ROOT}"
tar -xzf "${SOURCE_INPUT_ARCHIVE}" -C "${STAGING_ROOT}" --no-same-owner --no-same-permissions
EXECUTION_DIR="${STAGING_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
for required in \
  freeze_source.py OPTIMIZER_ZERO_LR_AUDIT_REPAIR_V8.json \
  optimizer_smoke_v8.sbatch submit_optimizer_smoke_v8_once.sh \
  train_v8.sbatch planner64_v8.sbatch submit_training64_v8_once.sh; do
  test -f "${EXECUTION_DIR}/${required}"
done

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/freeze_source.py" \
  --project-root "${STAGING_ROOT}" --output-root "${FREEZE_ROOT}"
test -f "${FREEZE_ROOT}/FREEZE_RECORD.json"
test -d "${FREEZE_ROOT}/source"
test -f "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_smact_split_v2.tar.gz"

mkdir "${RUN_ROOT}"
mkdir "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
cp -a "${FREEZE_ROOT}/source" "${RUN_ROOT}/source"
cp "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_smact_split_v2.tar.gz" \
  "${RUN_ROOT}/source_archive.tar.gz"
cp "${FREEZE_ROOT}/FREEZE_RECORD.json" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
cd "${RUN_ROOT}/source"
sha256sum -c SOURCE_SHA256.txt

sha256sum "${SOURCE_INPUT_ARCHIVE}" > "${RUN_ROOT}/status/transfer_inputs.sha256"
sha256sum "${RUN_ROOT}/source_archive.tar.gz" > "${RUN_ROOT}/status/source_archive.sha256"
sha256sum "${RUN_ROOT}/source/SOURCE_SHA256.txt" > "${RUN_ROOT}/status/source_inventory.sha256"

find "${RUN_ROOT}/source" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/source" -type d -exec chmod 500 {} +
chmod 400 "${RUN_ROOT}/source_archive.tar.gz" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
find "${STAGING_ROOT}" -type f -exec chmod 400 {} +
find "${STAGING_ROOT}" -type d -exec chmod 500 {} +
find "${FREEZE_ROOT}" -type f -exec chmod 400 {} +
find "${FREEZE_ROOT}" -type d -exec chmod 500 {} +
find "${TRANSFER_ROOT}" -type f -exec chmod 400 {} +
find "${TRANSFER_ROOT}" -type d -exec chmod 500 {} +
printf '%s\n' pass > "${RUN_ROOT}/status/source_bootstrap.status"
cat "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"

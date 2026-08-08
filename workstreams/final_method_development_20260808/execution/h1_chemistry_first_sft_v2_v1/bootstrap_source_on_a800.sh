#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
TRANSFER_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_transfer_input_v2"
STAGING_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_source_staging_v1"
FREEZE_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_source_freeze_v1"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_v1"
BUNDLE_INPUT_ROOT="${PROJECT_ROOT}/runs/20260808_smact4_400_runtime_bundle_input_v1"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SOURCE_INPUT_ARCHIVE="${1:?source input archive path}"
EXPECTED_SOURCE_INPUT_SHA256="${2:?source input archive SHA256}"
RUNTIME_BUNDLE_SOURCE="${3:?runtime bundle path}"
EXPECTED_RUNTIME_BUNDLE_SHA256=4ffac0ce561483fcacbb592cb9287b2e24bb4fbca67217396f7a2743a3de44bc

test -x "${LEGACY_PYTHON}"
test -d "${TRANSFER_ROOT}"
test -f "${SOURCE_INPUT_ARCHIVE}"
test -f "${RUNTIME_BUNDLE_SOURCE}"
for path in "${STAGING_ROOT}" "${FREEZE_ROOT}" "${RUN_ROOT}" "${BUNDLE_INPUT_ROOT}"; do
  test ! -e "${path}"
done
test "$(sha256sum "${SOURCE_INPUT_ARCHIVE}" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INPUT_SHA256}"
test "$(sha256sum "${RUNTIME_BUNDLE_SOURCE}" | cut -d' ' -f1)" = "${EXPECTED_RUNTIME_BUNDLE_SHA256}"

mkdir "${STAGING_ROOT}"
tar -xzf "${SOURCE_INPUT_ARCHIVE}" -C "${STAGING_ROOT}" --no-same-owner --no-same-permissions
EXECUTION_DIR="${STAGING_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
test -f "${EXECUTION_DIR}/freeze_source.py"
test -f "${EXECUTION_DIR}/SMACT4_RUNTIME_BUNDLE_FREEZE_RECORD.json"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/freeze_source.py" \
  --project-root "${STAGING_ROOT}" --output-root "${FREEZE_ROOT}"
test -f "${FREEZE_ROOT}/FREEZE_RECORD.json"
test -d "${FREEZE_ROOT}/source"
test -f "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_v1.tar.gz"

mkdir "${RUN_ROOT}"
mkdir "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
cp -a "${FREEZE_ROOT}/source" "${RUN_ROOT}/source"
cp "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_v1.tar.gz" \
  "${RUN_ROOT}/source_archive.tar.gz"
cp "${FREEZE_ROOT}/FREEZE_RECORD.json" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
cd "${RUN_ROOT}/source"
sha256sum -c SOURCE_SHA256.txt

mkdir "${BUNDLE_INPUT_ROOT}"
cp "${RUNTIME_BUNDLE_SOURCE}" \
  "${BUNDLE_INPUT_ROOT}/smact4_400_runtime_v1_bundle.tar.gz"
test "$(sha256sum "${BUNDLE_INPUT_ROOT}/smact4_400_runtime_v1_bundle.tar.gz" | cut -d' ' -f1)" \
  = "${EXPECTED_RUNTIME_BUNDLE_SHA256}"
sha256sum "${SOURCE_INPUT_ARCHIVE}" "${RUNTIME_BUNDLE_SOURCE}" \
  > "${RUN_ROOT}/status/transfer_inputs.sha256"
sha256sum "${RUN_ROOT}/source_archive.tar.gz" \
  > "${RUN_ROOT}/status/source_archive.sha256"
sha256sum "${RUN_ROOT}/source/SOURCE_SHA256.txt" \
  > "${RUN_ROOT}/status/source_inventory.sha256"

find "${RUN_ROOT}/source" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/source" -type d -exec chmod 500 {} +
chmod 400 "${RUN_ROOT}/source_archive.tar.gz" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
find "${BUNDLE_INPUT_ROOT}" -type f -exec chmod 400 {} +
chmod 500 "${BUNDLE_INPUT_ROOT}"
find "${STAGING_ROOT}" -type f -exec chmod 400 {} +
find "${STAGING_ROOT}" -type d -exec chmod 500 {} +
find "${FREEZE_ROOT}" -type f -exec chmod 400 {} +
find "${FREEZE_ROOT}" -type d -exec chmod 500 {} +
find "${TRANSFER_ROOT}" -type f -exec chmod 400 {} +
find "${TRANSFER_ROOT}" -type d -exec chmod 500 {} +
printf '%s\n' 'pass' > "${RUN_ROOT}/status/source_bootstrap.status"
cat "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"

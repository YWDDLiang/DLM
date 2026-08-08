#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
BUNDLE_INPUT_ROOT="${PROJECT_ROOT}/runs/20260808_smact4_400_runtime_bundle_input_v1"
BUNDLE_ARCHIVE="${BUNDLE_INPUT_ROOT}/smact4_400_runtime_v1_bundle.tar.gz"
RUNTIME_ROOT="${PROJECT_ROOT}/runs/20260808_smact4_400_runtime_v1"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_BUNDLE_SHA256=4ffac0ce561483fcacbb592cb9287b2e24bb4fbca67217396f7a2743a3de44bc

test -d "${RUN_ROOT}/logs"
test -d "${RUN_ROOT}/status"
test -d "${SOURCE_ROOT}"
test -x "${LEGACY_PYTHON}"
test -d "${BUNDLE_INPUT_ROOT}"
test -f "${BUNDLE_ARCHIVE}"
test ! -e "${RUNTIME_ROOT}"
test ! -e "$(dirname "${RUNTIME_ROOT}")/.${RUNTIME_ROOT##*/}.building"
observed_bundle_sha="$(sha256sum "${BUNDLE_ARCHIVE}" | cut -d' ' -f1)"
test "${observed_bundle_sha}" = "${EXPECTED_BUNDLE_SHA256}"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/prepare_shared_smact4_runtime.py" \
  --bundle-archive "${BUNDLE_ARCHIVE}" \
  --bundle-freeze-record "${EXECUTION_DIR}/SMACT4_RUNTIME_BUNDLE_FREEZE_RECORD.json" \
  --wrapper-source "${EXECUTION_DIR}/shared_smact4_python.sh" \
  --project-source-root "${SOURCE_ROOT}" \
  --output-root "${RUNTIME_ROOT}" \
  > "${RUN_ROOT}/logs/smact4_runtime_prepare.out" \
  2> "${RUN_ROOT}/logs/smact4_runtime_prepare.err"

test -f "${RUNTIME_ROOT}/terminal_report.json"
test -f "${RUNTIME_ROOT}/terminal_report.sha256"
test -f "${RUNTIME_ROOT}/_SUCCESS"
sha256sum "${RUNTIME_ROOT}/terminal_report.json" \
  > "${RUN_ROOT}/status/smact4_runtime_terminal.sha256"
printf '%s\n' 'pass' > "${RUN_ROOT}/status/smact4_runtime_prepare.status"
printf '%s\n' "${RUNTIME_ROOT}/python"

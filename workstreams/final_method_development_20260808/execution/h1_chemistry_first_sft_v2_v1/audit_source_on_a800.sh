#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
ISOLATED_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_isolated_archive_test_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SMACT4_PYTHON="${PROJECT_ROOT}/runs/20260808_smact4_400_runtime_v1/python"

test -d "${SOURCE_ROOT}"
test -f "${RUN_ROOT}/source_archive.tar.gz"
test -x "${LEGACY_PYTHON}"
test -x "${SMACT4_PYTHON}"
test ! -e "${ISOLATED_ROOT}"
SOURCE_INVENTORY_SHA256="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${SOURCE_ROOT}"
"${LEGACY_PYTHON}" -m unittest \
  tests.test_h1_chemistry_first_sft \
  tests.test_h1_nocharge_sft_tokenizer_audit \
  tests.test_h1_nocharge_planner_gate \
  tests.test_h1_nocharge_ion_aux \
  workstreams.final_method_development_20260808.execution.h1_chemistry_first_sft_v2_v1.test_protocol \
  > "${RUN_ROOT}/logs/a800_source_tests.out" \
  2> "${RUN_ROOT}/logs/a800_source_tests.err"

mkdir "${ISOLATED_ROOT}"
tar -xzf "${RUN_ROOT}/source_archive.tar.gz" -C "${ISOLATED_ROOT}" \
  --no-same-owner --no-same-permissions
cd "${ISOLATED_ROOT}"
sha256sum -c SOURCE_SHA256.txt
export PYTHONPATH="${ISOLATED_ROOT}"
"${LEGACY_PYTHON}" -m unittest \
  tests.test_h1_chemistry_first_sft \
  tests.test_h1_nocharge_sft_tokenizer_audit \
  tests.test_h1_nocharge_planner_gate \
  tests.test_h1_nocharge_ion_aux \
  workstreams.final_method_development_20260808.execution.h1_chemistry_first_sft_v2_v1.test_protocol \
  > "${RUN_ROOT}/logs/a800_isolated_tests.out" \
  2> "${RUN_ROOT}/logs/a800_isolated_tests.err"

mkdir "${RUN_ROOT}/preflight"
export PYTHONPATH="${SOURCE_ROOT}"
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
  --expected-source-inventory-sha256 "${SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/a800_runtime_preflight.json" \
  > "${RUN_ROOT}/logs/a800_runtime_preflight.out" \
  2> "${RUN_ROOT}/logs/a800_runtime_preflight.err"

sha256sum "${RUN_ROOT}/preflight/a800_runtime_preflight.json" \
  > "${RUN_ROOT}/preflight/a800_runtime_preflight.sha256"
find "${ISOLATED_ROOT}" -type f -exec chmod 400 {} +
find "${ISOLATED_ROOT}" -type d -exec chmod 500 {} +
printf '%s\n' 'pass' > "${RUN_ROOT}/status/a800_source_audit.status"
printf '%s\n' "${SOURCE_INVENTORY_SHA256}"

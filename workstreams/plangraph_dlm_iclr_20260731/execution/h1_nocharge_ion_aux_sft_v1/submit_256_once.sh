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

test -f "${RUN_ROOT}/submission_record.json"
test -f "${RUN_ROOT}/planner64/_SUCCESS"
test -f "${RUN_ROOT}/planner64/terminal/terminal_report.json"
test ! -e "${RUN_ROOT}/submission256_record.json"
test ! -e "${RUN_ROOT}/planner256"
mkdir "${RUN_ROOT}/.submit_planner256_lock"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"
PRIOR64_TERMINAL_SHA="$(sha256sum "${RUN_ROOT}/planner64/terminal/terminal_report.json" | cut -d' ' -f1)"
test "$("${LEGACY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RUN_ROOT}/planner64/terminal/terminal_report.json")" = planner_gate_pass

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/preflight.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${EXECUTION_DIR}/CONFIG.json" \
  --authorization "${EXECUTION_DIR}/AUTHORIZATION.json" \
  --ledger64 "${EXECUTION_DIR}/LEDGER64.json" \
  --ledger256 "${EXECUTION_DIR}/LEDGER256.json" \
  --legacy-python "${LEGACY_PYTHON}" --smact4-python "${SMACT4_PYTHON}" \
  --model-path "${MODEL_PATH}" --p0-adapter-path "${P0_ADAPTER}" \
  --mp20-dir "${MP20_DIR}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
for partition in gpu normal; do
  printf '%s\n' "${partition_snapshot}" \
    | awk -F'|' -v wanted="${partition}" '$1 == wanted {found=1} END {exit found ? 0 : 1}'
done
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_planner256.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_planner256.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON},SMACT4_PYTHON=${SMACT4_PYTHON},EXPECTED_LEDGER_SHA256=${LEDGER256_SHA}"
PLANNER_JOB_ID="$(sbatch --parsable --array=0-2%2 --export="${common_export}" "${EXECUTION_DIR}/planner256.sbatch")"
printf '%s\n' "${PLANNER_JOB_ID}" > "${RUN_ROOT}/status/submitted_planner256_job_id.txt"
ASSEMBLY_JOB_ID="$(sbatch --parsable --dependency=afterany:"${PLANNER_JOB_ID}" --export="${common_export}" "${EXECUTION_DIR}/assemble256.sbatch")"
printf '%s\n' "${ASSEMBLY_JOB_ID}" > "${RUN_ROOT}/status/submitted_assemble256_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON SMACT4_PYTHON SINFO_SHA
export PLANNER_JOB_ID ASSEMBLY_JOB_ID PRIOR64_TERMINAL_SHA
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage planner256 --output "${RUN_ROOT}/submission256_record.json"
sha256sum "${RUN_ROOT}/submission256_record.json" > "${RUN_ROOT}/submission256_record.sha256"
printf 'planner256=%s\nassemble256=%s\n' "${PLANNER_JOB_ID}" "${ASSEMBLY_JOB_ID}"

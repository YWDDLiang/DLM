#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
FROZEN_AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected SOURCE_SHA256.txt digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"
LOCAL_SMACT4_AUDIT_ROOT="${3:?local raw256 SMACT4 audit root}"
EXPECTED_LOCAL_SMACT4_AUDIT_MANIFEST_SHA256="${4:?local raw256 audit manifest digest}"

test -f "${RUN_ROOT}/submission256_record.json"
test -f "${RUN_ROOT}/submission256_record.sha256"
sha256sum -c "${RUN_ROOT}/submission256_record.sha256"
test "${LOCAL_SMACT4_AUDIT_ROOT}" = "${FROZEN_AUDIT_ROOT}"
test -f "${LOCAL_SMACT4_AUDIT_ROOT}/MANIFEST.json"
test -f "${LOCAL_SMACT4_AUDIT_ROOT}/_SUCCESS"
test "$(sha256sum "${LOCAL_SMACT4_AUDIT_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = "${EXPECTED_LOCAL_SMACT4_AUDIT_MANIFEST_SHA256}"
test ! -e "${RUN_ROOT}/assembly256_submission_record.json"
test ! -e "${RUN_ROOT}/planner256/terminal"
test ! -e "${RUN_ROOT}/planner256/_SUCCESS"
test ! -e "${RUN_ROOT}/planner256/_SCIENTIFIC_STOP"
test ! -e "${RUN_ROOT}/planner256/_FAILED"
mkdir "${RUN_ROOT}/.submit_science256_assembly_lock"

test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"
PRIOR_GENERATION_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/submission256_record.json" | cut -d' ' -f1)"
EXPECTED_CANDIDATES="$("${LEGACY_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d["candidate_list"]))' "${RUN_ROOT}/submission256_record.json")"
PLANNER_JOB_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/status/submitted_planner256_job_id.txt")"
case "${PLANNER_JOB_ID}" in ''|*[!0-9]*) echo "invalid planner256 job id: ${PLANNER_JOB_ID}" >&2; exit 3 ;; esac
expected_tasks=(0)
case "${EXPECTED_CANDIDATES}" in
  sft_v2) expected_tasks+=(1) ;;
  sft_v2_c) expected_tasks+=(2) ;;
  sft_v2,sft_v2_c) expected_tasks+=(1 2) ;;
  *) echo "invalid planner256 candidate list: ${EXPECTED_CANDIDATES}" >&2; exit 3 ;;
esac
sacct -n -X -j "${PLANNER_JOB_ID}" -o JobIDRaw,State,ExitCode -P \
  > "${RUN_ROOT}/status/sacct_planner256_before_assembly.txt"
for task in "${expected_tasks[@]}"; do
  expected_id="${PLANNER_JOB_ID}_${task}"
  state="$(awk -F'|' -v wanted="${expected_id}" '$1 == wanted {print $2}' "${RUN_ROOT}/status/sacct_planner256_before_assembly.txt")"
  test "$(printf '%s\n' "${state}" | sed '/^$/d' | wc -l)" -eq 1
  case "${state%%+*}" in
    COMPLETED|FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|CANCELLED|PREEMPTED|BOOT_FAIL|DEADLINE) ;;
    *) echo "planner256 task not terminal: ${expected_id} state=${state}" >&2; exit 3 ;;
  esac
done
GENERATION_SACCT_SHA="$(sha256sum "${RUN_ROOT}/status/sacct_planner256_before_assembly.txt" | cut -d' ' -f1)"
AUDITED_ARMS="$("${LEGACY_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass" and d["stage"]==256 and d["denominator"]==256; print(",".join(d["arms"]))' "${LOCAL_SMACT4_AUDIT_ROOT}/MANIFEST.json")"
case "${AUDITED_ARMS}" in p0|p0,sft_v2|p0,sft_v2_c|p0,sft_v2,sft_v2_c) ;; *) exit 3 ;; esac

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/preflight.py" \
  --source-root "${SOURCE_ROOT}" --config "${EXECUTION_DIR}/CONFIG.json" \
  --authorization "${EXECUTION_DIR}/AUTHORIZATION.json" \
  --ledger64 "${EXECUTION_DIR}/LEDGER64.json" --ledger256 "${EXECUTION_DIR}/LEDGER256.json" \
  --legacy-python "${LEGACY_PYTHON}" --model-path "${MODEL_PATH}" \
  --p0-adapter-path "${P0_ADAPTER}" --mp20-dir "${MP20_DIR}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/preflight256_assembly_report.json"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/preflight256_assembly_report.json" | cut -d' ' -f1)"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "normal" && $2 == "up" {found=1} END {exit found ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_science256_assembly.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' > "${RUN_ROOT}/status/squeue_before_science256_assembly.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_science256_assembly.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_science256_assembly.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON},EXPECTED_LEDGER_SHA256=${LEDGER256_SHA},EXPECTED_CANDIDATES=${EXPECTED_CANDIDATES},LOCAL_SMACT4_AUDIT_ROOT=${LOCAL_SMACT4_AUDIT_ROOT},EXPECTED_LOCAL_SMACT4_AUDIT_MANIFEST_SHA256=${EXPECTED_LOCAL_SMACT4_AUDIT_MANIFEST_SHA256},AUDITED_ARMS=${AUDITED_ARMS}"
ASSEMBLY_JOB_ID="$(sbatch --parsable --export="${common_export}" "${EXECUTION_DIR}/assemble256.sbatch")"
printf '%s\n' "${ASSEMBLY_JOB_ID}" > "${RUN_ROOT}/status/submitted_assemble256_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export ASSEMBLY_JOB_ID PRIOR_GENERATION_SUBMISSION_SHA LOCAL_SMACT4_AUDIT_ROOT AUDITED_ARMS EXPECTED_CANDIDATES
export GENERATION_SACCT_SHA
export LOCAL_SMACT4_AUDIT_MANIFEST_SHA="${EXPECTED_LOCAL_SMACT4_AUDIT_MANIFEST_SHA256}"
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage planner256_assembly --output "${RUN_ROOT}/assembly256_submission_record.json"
sha256sum "${RUN_ROOT}/assembly256_submission_record.json" > "${RUN_ROOT}/assembly256_submission_record.sha256"
printf 'assemble256=%s\naudited_arms=%s\n' "${ASSEMBLY_JOB_ID}" "${AUDITED_ARMS}"

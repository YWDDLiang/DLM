#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7"
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
test -f "${RUN_ROOT}/engineering_submission_record.json"
test -f "${RUN_ROOT}/engineering_submission_record.sha256"
sha256sum -c "${RUN_ROOT}/engineering_submission_record.sha256"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test -f "${RUN_ROOT}/status/data_SUCCESS"
test -f "${RUN_ROOT}/status/smoke_sft_v2_SUCCESS"
test -f "${RUN_ROOT}/status/smoke_sft_v2_c_SUCCESS"
test -f "${RUN_ROOT}/status/submitted_smoke_job_id.txt"
test -f "${RUN_ROOT}/identity_probe_submission_record.json"
sha256sum -c "${RUN_ROOT}/identity_probe_submission_record.sha256"
test -f "${RUN_ROOT}/probe/real_p0_identity_report.json"
sha256sum -c "${RUN_ROOT}/probe/real_p0_identity_report.sha256"
test -f "${RUN_ROOT}/probe/real_p0_identity_gate.json"
sha256sum -c "${RUN_ROOT}/probe/real_p0_identity_gate.sha256"
test -x "${LEGACY_PYTHON}"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/training"
test ! -e "${RUN_ROOT}/planner64"
mkdir "${RUN_ROOT}/.submit_science64_generation_lock"

observed_source_sha="$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
observed_archive_sha="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"
test "${observed_source_sha}" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "${observed_archive_sha}" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"
PRIOR_ENGINEERING_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/engineering_submission_record.json" | cut -d' ' -f1)"

SMOKE_JOB_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/status/submitted_smoke_job_id.txt")"
case "${SMOKE_JOB_ID}" in ''|*[!0-9]*) echo "invalid smoke job id: ${SMOKE_JOB_ID}" >&2; exit 3 ;; esac
sacct -n -X -j "${SMOKE_JOB_ID}" -o JobID,State,ExitCode -P > "${RUN_ROOT}/status/sacct_smoke_before_science64.txt"
for task in 0 1; do
  expected_id="${SMOKE_JOB_ID}_${task}"
  matches="$(awk -F'|' -v wanted="${expected_id}" '$1 == wanted {print $2 "|" $3}' "${RUN_ROOT}/status/sacct_smoke_before_science64.txt")"
  test "${matches}" = "COMPLETED|0:0"
done

test ! -e "${RUN_ROOT}/preflight/identity_probe_admission_before_training.json"
"${LEGACY_PYTHON}" scripts/a800/validate_h1_peft_identity_gate_v1.py probe \
  --report "${RUN_ROOT}/probe/real_p0_identity_report.json" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/identity_probe_admission_before_training.json"
IDENTITY_PROBE_ADMISSION_SHA="$(sha256sum "${RUN_ROOT}/preflight/identity_probe_admission_before_training.json" | cut -d' ' -f1)"
for candidate in sft_v2 sft_v2_c; do
  test -f "${RUN_ROOT}/smoke/${candidate}/identity_gate_report.json"
  sha256sum -c "${RUN_ROOT}/smoke/${candidate}/identity_gate_report.sha256"
  admission="${RUN_ROOT}/preflight/smoke_admission_${candidate}_before_training.json"
  test ! -e "${admission}"
  "${LEGACY_PYTHON}" scripts/a800/validate_h1_peft_identity_gate_v1.py smoke \
    --smoke-dir "${RUN_ROOT}/smoke/${candidate}" \
    --candidate "${candidate}" \
    --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
    --output "${admission}"
done
SMOKE_ADMISSION_SFT_V2_SHA="$(sha256sum "${RUN_ROOT}/preflight/smoke_admission_sft_v2_before_training.json" | cut -d' ' -f1)"
SMOKE_ADMISSION_SFT_V2_C_SHA="$(sha256sum "${RUN_ROOT}/preflight/smoke_admission_sft_v2_c_before_training.json" | cut -d' ' -f1)"

test -d "${RUN_ROOT}/preflight"
test ! -e "${RUN_ROOT}/preflight/preflight_science64_generation_report.json"
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
  --output "${RUN_ROOT}/preflight/preflight_science64_generation_report.json"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/preflight_science64_generation_report.json" | cut -d' ' -f1)"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
for partition in gpu gpu_long; do
  printf '%s\n' "${partition_snapshot}" | awk -F'|' -v wanted="${partition}" '$1 == wanted && $2 == "up" {found=1} END {exit found ? 0 : 1}'
done
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_science64_generation.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' > "${RUN_ROOT}/status/squeue_before_science64_generation.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_science64_generation.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_science64_generation.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON}"
TRAIN_JOB_ID="$(sbatch --parsable --array=0-1%2 --export="${common_export}" "${EXECUTION_DIR}/train.sbatch")"
printf '%s\n' "${TRAIN_JOB_ID}" > "${RUN_ROOT}/status/submitted_train_job_id.txt"
PLANNER_JOB_ID="$(sbatch --parsable --array=0-2%2 --dependency=afterany:"${TRAIN_JOB_ID}" --export="${common_export},EXPECTED_LEDGER_SHA256=${LEDGER64_SHA}" "${EXECUTION_DIR}/planner64.sbatch")"
printf '%s\n' "${PLANNER_JOB_ID}" > "${RUN_ROOT}/status/submitted_planner64_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export TRAIN_JOB_ID PLANNER_JOB_ID PRIOR_ENGINEERING_SUBMISSION_SHA
export IDENTITY_PROBE_ADMISSION_SHA SMOKE_ADMISSION_SFT_V2_SHA SMOKE_ADMISSION_SFT_V2_C_SHA
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage planner64_generation --output "${RUN_ROOT}/submission_record.json"
sha256sum "${RUN_ROOT}/submission_record.json" > "${RUN_ROOT}/submission_record.sha256"
printf 'train=%s\nplanner64=%s\n' "${TRAIN_JOB_ID}" "${PLANNER_JOB_ID}"

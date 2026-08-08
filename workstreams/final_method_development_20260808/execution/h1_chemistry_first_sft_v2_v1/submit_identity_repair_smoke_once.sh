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
LOCAL_SMACT4_WITNESS_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact4_witness_input_v1"
EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256=d21698e29664c607541d7ab644250e93e18cfb2a0cd03d1687a270b42c8ccd32
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected repaired source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected repaired source archive digest}"

test -d "${SOURCE_ROOT}"
test -x "${LEGACY_PYTHON}"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test -f "${RUN_ROOT}/status/data_SUCCESS"
test -f "${RUN_ROOT}/DATA_REUSE_RECORD.json"
sha256sum -c "${RUN_ROOT}/DATA_REUSE_RECORD.sha256"
test -f "${RUN_ROOT}/snapshot_submission_record.json"
sha256sum -c "${RUN_ROOT}/snapshot_submission_record.sha256"
test -f "${RUN_ROOT}/identity_probe_submission_record.json"
sha256sum -c "${RUN_ROOT}/identity_probe_submission_record.sha256"
test -f "${RUN_ROOT}/status/identity_probe_SUCCESS"
test -f "${RUN_ROOT}/status/submitted_identity_probe_job_id.txt"
test -f "${RUN_ROOT}/probe/real_p0_identity_report.json"
test -f "${RUN_ROOT}/probe/real_p0_identity_gate.json"
sha256sum -c "${RUN_ROOT}/probe/real_p0_identity_report.sha256"
sha256sum -c "${RUN_ROOT}/probe/real_p0_identity_gate.sha256"
test "$(sha256sum "${LOCAL_SMACT4_WITNESS_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = "${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}"
test ! -e "${RUN_ROOT}/engineering_submission_record.json"
test ! -e "${RUN_ROOT}/smoke"
test ! -e "${RUN_ROOT}/training"
test ! -e "${RUN_ROOT}/planner64"

test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
PROBE_JOB_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/status/submitted_identity_probe_job_id.txt")"
case "${PROBE_JOB_ID}" in ''|*[!0-9]*) echo "invalid identity probe job id: ${PROBE_JOB_ID}" >&2; exit 3 ;; esac
sacct -n -X -j "${PROBE_JOB_ID}" -o JobIDRaw,State,ExitCode -P \
  > "${RUN_ROOT}/status/sacct_identity_probe_before_smoke.txt"
test "$(awk -F'|' -v wanted="${PROBE_JOB_ID}" '$1 == wanted {print $2 "|" $3}' "${RUN_ROOT}/status/sacct_identity_probe_before_smoke.txt")" = "COMPLETED|0:0"
test ! -e "${RUN_ROOT}/preflight/identity_probe_admission_before_smoke.json"
"${LEGACY_PYTHON}" scripts/a800/validate_h1_peft_identity_gate_v1.py probe \
  --report "${RUN_ROOT}/probe/real_p0_identity_report.json" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/identity_probe_admission_before_smoke.json"
IDENTITY_PROBE_REPORT_SHA="$(sha256sum "${RUN_ROOT}/probe/real_p0_identity_report.json" | cut -d' ' -f1)"
IDENTITY_PROBE_GATE_SHA="$(sha256sum "${RUN_ROOT}/probe/real_p0_identity_gate.json" | cut -d' ' -f1)"
IDENTITY_PROBE_ADMISSION_SHA="$(sha256sum "${RUN_ROOT}/preflight/identity_probe_admission_before_smoke.json" | cut -d' ' -f1)"
mkdir "${RUN_ROOT}/.submit_identity_repair_smoke_lock"
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"

test -d "${RUN_ROOT}/preflight"
test ! -e "${RUN_ROOT}/preflight/preflight_identity_repair_smoke_report.json"
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
  --output "${RUN_ROOT}/preflight/preflight_identity_repair_smoke_report.json"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/preflight_identity_repair_smoke_report.json" | cut -d' ' -f1)"

partition_snapshot="$(sinfo -h -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "gpu" && $2 == "up" {found=1} END {exit found ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_identity_repair_smoke.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' > "${RUN_ROOT}/status/squeue_before_identity_repair_smoke.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_identity_repair_smoke.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_identity_repair_smoke.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON}"
SMOKE_JOB_ID="$(sbatch --parsable --array=0-1%2 --export="${common_export}" "${EXECUTION_DIR}/smoke.sbatch")"
printf '%s\n' "${SMOKE_JOB_ID}" > "${RUN_ROOT}/status/submitted_smoke_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export DATA_JOB_ID=31035 SMOKE_JOB_ID LOCAL_SMACT4_WITNESS_ROOT
export LOCAL_SMACT4_WITNESS_MANIFEST_SHA="${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}"
export PRIOR_SNAPSHOT_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/snapshot_submission_record.json" | cut -d' ' -f1)"
export PRIOR_IDENTITY_PROBE_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/identity_probe_submission_record.json" | cut -d' ' -f1)"
export IDENTITY_PROBE_REPORT_SHA IDENTITY_PROBE_GATE_SHA IDENTITY_PROBE_ADMISSION_SHA
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage engineering_smoke --output "${RUN_ROOT}/engineering_submission_record.json"
sha256sum "${RUN_ROOT}/engineering_submission_record.json" > "${RUN_ROOT}/engineering_submission_record.sha256"
printf 'smoke=%s\n' "${SMOKE_JOB_ID}"

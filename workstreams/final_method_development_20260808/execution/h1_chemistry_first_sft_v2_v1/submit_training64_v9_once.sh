#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_override_v9"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"

test -d "${SOURCE_ROOT}"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test -f "${RUN_ROOT}/status/source_delta_SUCCESS"
test -f "${RUN_ROOT}/status/data_SUCCESS"
test -f "${RUN_ROOT}/status/optimizer_smoke_reuse_SUCCESS"
test -f "${RUN_ROOT}/status/optimizer_smoke_sft_v2_SUCCESS"
test -f "${RUN_ROOT}/status/optimizer_smoke_sft_v2_c_SUCCESS"
sha256sum -c "${RUN_ROOT}/V9_PREPARATION_RECORD.sha256"
sha256sum -c "${RUN_ROOT}/SOURCE_DELTA_AUDIT.sha256"
sha256sum -c "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.sha256"
sha256sum -c "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.sha256"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/training"
test ! -e "${RUN_ROOT}/planner64"
mkdir "${RUN_ROOT}/.submit_training64_v9_lock"

test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
LEDGER64_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)"
LEDGER256_SHA="$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)"
PRIOR_ENGINEERING_SUBMISSION_SHA="$(sha256sum "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json" | cut -d' ' -f1)"
SOURCE_DELTA_AUDIT_SHA="$(sha256sum "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" | cut -d' ' -f1)"

OPTIMIZER_SMOKE_JOB_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/status/submitted_optimizer_smoke_job_id.txt")"
case "${OPTIMIZER_SMOKE_JOB_ID}" in ''|*[!0-9]*) echo "invalid optimizer smoke job id" >&2; exit 3 ;; esac
sacct -n -X -j "${OPTIMIZER_SMOKE_JOB_ID}" -o JobID,State,ExitCode -P \
  > "${RUN_ROOT}/status/sacct_reused_optimizer_smoke_before_training_v9.txt"
for task in 0 1; do
  expected_id="${OPTIMIZER_SMOKE_JOB_ID}_${task}"
  matches="$(awk -F'|' -v wanted="${expected_id}" '$1 == wanted {print $2 "|" $3}' \
    "${RUN_ROOT}/status/sacct_reused_optimizer_smoke_before_training_v9.txt")"
  test "${matches}" = "COMPLETED|0:0"
done

for candidate in sft_v2 sft_v2_c; do
  report="${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.json"
  sha256sum -c "${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.sha256"
  admission="${RUN_ROOT}/preflight/optimizer_smoke_admission_${candidate}_before_training.json"
  test ! -e "${admission}"
  "${LEGACY_PYTHON}" - "${report}" "${candidate}" "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
    "${SOURCE_DELTA_AUDIT_SHA}" "${admission}" <<'PY'
import hashlib, json, sys
source, candidate, source_sha, delta_sha, output = sys.argv[1:]
report = json.load(open(source, encoding="utf-8"))
failures = []
if report.get("status") != "pass": failures.append("status")
if report.get("candidate") != candidate: failures.append("candidate")
if report.get("optimizer_updates") != 2: failures.append("updates")
if report.get("microbatch_count") != 16: failures.append("microbatches")
if report.get("full_training_total_updates") != 4505: failures.append("total_updates")
if report.get("full_training_warmup_steps") != 135: failures.append("warmup")
if report.get("scientific_checkpoint_saved") is not False: failures.append("checkpoint")
if report.get("generation") is not False: failures.append("generation")
if report.get("smact4_executed_on_a800") is not False: failures.append("smact4")
if report.get("failures") != []: failures.append("reported_failures")
audits = report.get("optimizer_step_audits") or []
if len(audits) != 2 or not all(item.get("passed") is True for item in audits):
    failures.append("audits")
payload = {
    "schema": "h1_chemistry_first_optimizer_smoke_admission_v2",
    "candidate": candidate,
    "source_inventory_sha256": source_sha,
    "source_delta_audit_sha256": delta_sha,
    "optimizer_smoke_report_sha256": hashlib.sha256(open(source, "rb").read()).hexdigest(),
    "reused_from_parent_v8": True,
    "partition_only_change": True,
    "optimizer_smoke_repeated": False,
    "failures": failures,
    "passed": not failures,
}
open(output, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if failures:
    raise SystemExit(f"optimizer smoke admission failed: {failures}")
PY
done
OPTIMIZER_SMOKE_ADMISSION_SFT_V2_SHA="$(sha256sum "${RUN_ROOT}/preflight/optimizer_smoke_admission_sft_v2_before_training.json" | cut -d' ' -f1)"
OPTIMIZER_SMOKE_ADMISSION_SFT_V2_C_SHA="$(sha256sum "${RUN_ROOT}/preflight/optimizer_smoke_admission_sft_v2_c_before_training.json" | cut -d' ' -f1)"

sha256sum -c "${RUN_ROOT}/preflight/a800_runtime_preflight.sha256"
PREFLIGHT_SHA="$(sha256sum "${RUN_ROOT}/preflight/a800_runtime_preflight.json" | cut -d' ' -f1)"
partition_snapshot="$(sinfo -h -p gpu -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "gpu" && $2 == "up" {found=1} END {exit found ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_training64_v9.txt"
scontrol show partition gpu -o > "${RUN_ROOT}/status/scontrol_gpu_before_training64_v9.txt"
"${LEGACY_PYTHON}" - "${RUN_ROOT}/status/scontrol_gpu_before_training64_v9.txt" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"\bMaxTime=([^\s]+)", text)
if not match:
    raise SystemExit("gpu partition MaxTime unavailable")
value = match.group(1)
if value.upper() == "INFINITE":
    raise SystemExit(0)
days = 0
clock = value
if "-" in value:
    day_text, clock = value.split("-", 1)
    days = int(day_text)
parts = [int(part) for part in clock.split(":")]
if len(parts) == 3:
    hours, minutes, seconds = parts
elif len(parts) == 2:
    hours, minutes = parts
    seconds = 0
else:
    raise SystemExit(f"unsupported MaxTime={value}")
total = days * 86400 + hours * 3600 + minutes * 60 + seconds
if total < 30 * 3600:
    raise SystemExit(f"gpu partition MaxTime={value} is below 30 hours")
PY
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' \
  > "${RUN_ROOT}/status/squeue_before_training64_v9.txt"
SINFO_SHA="$(sha256sum "${RUN_ROOT}/status/sinfo_before_training64_v9.txt" | cut -d' ' -f1)"
SQUEUE_SHA="$(sha256sum "${RUN_ROOT}/status/squeue_before_training64_v9.txt" | cut -d' ' -f1)"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},LEGACY_PYTHON=${LEGACY_PYTHON}"
TRAIN_JOB_ID="$(sbatch --parsable --array=0-1%2 --export="${common_export}" "${EXECUTION_DIR}/train_v9.sbatch")"
printf '%s\n' "${TRAIN_JOB_ID}" > "${RUN_ROOT}/status/submitted_train_job_id.txt"
PLANNER_JOB_ID="$(sbatch --parsable --array=0-2%2 --dependency=afterany:"${TRAIN_JOB_ID}" \
  --export="${common_export},EXPECTED_LEDGER_SHA256=${LEDGER64_SHA}" "${EXECUTION_DIR}/planner64_v9.sbatch")"
printf '%s\n' "${PLANNER_JOB_ID}" > "${RUN_ROOT}/status/submitted_planner64_job_id.txt"

export SOURCE_INVENTORY_SHA="${EXPECTED_SOURCE_INVENTORY_SHA256}"
export ARCHIVE_SHA="${EXPECTED_ARCHIVE_SHA256}"
export LEDGER64_SHA LEDGER256_SHA LEGACY_PYTHON PREFLIGHT_SHA SINFO_SHA SQUEUE_SHA
export TRAIN_JOB_ID PLANNER_JOB_ID PRIOR_ENGINEERING_SUBMISSION_SHA
export OPTIMIZER_SMOKE_ADMISSION_SFT_V2_SHA OPTIMIZER_SMOKE_ADMISSION_SFT_V2_C_SHA
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_submission_record.py" \
  --stage planner64_generation --output "${RUN_ROOT}/submission_record.json"
sha256sum "${RUN_ROOT}/submission_record.json" > "${RUN_ROOT}/submission_record.sha256"

export RUN_ROOT TRAIN_JOB_ID PLANNER_JOB_ID SOURCE_INVENTORY_SHA EXPECTED_ARCHIVE_SHA256
export SOURCE_DELTA_AUDIT_SHA
"${LEGACY_PYTHON}" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema": "h1_chemistry_first_v9_gpu_partition_submission_v1",
    "status": "submitted",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "jobs": {"training": os.environ["TRAIN_JOB_ID"], "planner64": os.environ["PLANNER_JOB_ID"]},
    "training_partition": "gpu",
    "planner_generation_partition": "gpu",
    "training_walltime": "30:00:00",
    "parent_cancelled_jobs": ["31136", "31137"],
    "source_inventory_sha256": os.environ["SOURCE_INVENTORY_SHA"],
    "source_archive_sha256": os.environ["EXPECTED_ARCHIVE_SHA256"],
    "source_delta_audit_sha256": os.environ["SOURCE_DELTA_AUDIT_SHA"],
    "restart_from_protected_p0": True,
    "resume_cancelled_state": False,
    "training_math_or_science_change": False,
    "smact4_executed_on_a800": False,
}
(root / "GPU_PARTITION_SUBMISSION_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
sha256sum "${RUN_ROOT}/GPU_PARTITION_SUBMISSION_RECORD.json" \
  > "${RUN_ROOT}/GPU_PARTITION_SUBMISSION_RECORD.sha256"
printf 'train=%s\nplanner64=%s\n' "${TRAIN_JOB_ID}" "${PLANNER_JOB_ID}"

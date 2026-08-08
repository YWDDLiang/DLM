#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="${ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
SOURCE="${RUN}/source"
EXECUTION="${SOURCE}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_SHA256="${1:?expected source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected source archive digest}"

test "$(cat "${RUN}/status/a800_source_audit.status")" = pass
test -f "${RUN}/status/data_SUCCESS"
test -f "${RUN}/status/optimizer_smoke_reuse_SUCCESS"
test -f "${RUN}/status/optimizer_smoke_sft_v2_SUCCESS"
test -f "${RUN}/status/optimizer_smoke_sft_v2_c_SUCCESS"
sha256sum -c "${RUN}/V10_PREPARATION_RECORD.sha256"
sha256sum -c "${RUN}/preflight/a800_runtime_preflight_v10.sha256"
test "$(sha256sum "${SOURCE}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_SHA256}"
test "$(sha256sum "${RUN}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
test ! -e "${RUN}/submission_record.json"
test ! -e "${RUN}/training"
test ! -e "${RUN}/planner64"
mkdir "${RUN}/.submit_training64_v10_lock"
(cd "${SOURCE}" && sha256sum -c SOURCE_SHA256.txt)

partition_snapshot="$(sinfo -h -p gpu -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' '$1 == "gpu" && $2 == "up" {ok=1} END {exit ok ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN}/status/sinfo_before_training64_v10.txt"
scontrol show partition gpu -o > "${RUN}/status/scontrol_gpu_before_training64_v10.txt"
grep -Eq 'MaxTime=(INFINITE|[2-9]-[0-9]{2}:[0-9]{2}:[0-9]{2}|1-(0[6-9]|1[0-9]|2[0-3]):[0-9]{2}:[0-9]{2})' \
  "${RUN}/status/scontrol_gpu_before_training64_v10.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' \
  > "${RUN}/status/squeue_before_training64_v10.txt"

LEDGER64_SHA="$(sha256sum "${EXECUTION}/LEDGER64.json" | cut -d' ' -f1)"
common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_SHA256},LEGACY_PYTHON=${PYTHON}"
TRAIN_JOB="$(sbatch --parsable --array=0-1%2 --export="${common_export}" "${EXECUTION}/train_v10.sbatch")"
printf '%s\n' "${TRAIN_JOB}" > "${RUN}/status/submitted_train_job_id.txt"
PLANNER_JOB="$(sbatch --parsable --array=0-2%2 --dependency=afterany:"${TRAIN_JOB}" \
  --export="${common_export},EXPECTED_LEDGER_SHA256=${LEDGER64_SHA}" "${EXECUTION}/planner64_v10.sbatch")"
printf '%s\n' "${PLANNER_JOB}" > "${RUN}/status/submitted_planner64_job_id.txt"

"${PYTHON}" - "${RUN}/submission_record.json" "${TRAIN_JOB}" "${PLANNER_JOB}" \
  "${EXPECTED_SOURCE_SHA256}" "${EXPECTED_ARCHIVE_SHA256}" "${LEDGER64_SHA}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
output = Path(sys.argv[1])
record = {
    "schema": "h1_chemistry_first_v10_submission_record_v1",
    "status": "complete",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "jobs": {"training": sys.argv[2], "planner64": sys.argv[3]},
    "dependency": f"afterany:{sys.argv[2]}",
    "training_array": "0-1%2", "planner64_array": "0-2%2",
    "training_partition": "gpu", "planner64_partition": "gpu",
    "source_inventory_sha256": sys.argv[4], "source_archive_sha256": sys.argv[5],
    "ledger64_sha256": sys.argv[6],
    "restart_from_protected_p0": True, "resume_cancelled_state": False,
    "repair_scope": "cancellation evidence parser only",
    "automatic_downstream": False, "automatic_rl": False,
    "smact4_executed_on_a800": False,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
sha256sum "${RUN}/submission_record.json" > "${RUN}/submission_record.sha256"
printf 'train=%s\nplanner64=%s\n' "${TRAIN_JOB}" "${PLANNER_JOB}"

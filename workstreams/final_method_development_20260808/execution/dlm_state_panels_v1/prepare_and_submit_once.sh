#!/bin/bash
set -Eeuo pipefail
umask 077

if [ "$#" -ne 1 ]; then
  echo "usage: $0 SOURCE_ARCHIVE" >&2
  exit 2
fi

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v1"
ARCHIVE="$(readlink -f "$1")"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

record_prepare_failure() {
  rc=$?
  if [ "${rc}" -ne 0 ] && [ -d "${RUN_ROOT}/status" ]; then
    printf '%s\n' "${rc}" > "${RUN_ROOT}/status/PREPARE_FAILED"
  fi
  exit "${rc}"
}
trap record_prepare_failure EXIT

test -f "${ARCHIVE}"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/source" "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
cp "${ARCHIVE}" "${RUN_ROOT}/source.tar"
tar -xf "${RUN_ROOT}/source.tar" -C "${RUN_ROOT}/source"

SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_state_panels_v1"
test -f "${EXECUTION_DIR}/SOURCE_FILES.txt"
cd "${SOURCE_ROOT}"
while IFS= read -r relative; do
  test -n "${relative}" || continue
  test -f "${relative}"
  sha256sum "${relative}"
done < "${EXECUTION_DIR}/SOURCE_FILES.txt" > "${RUN_ROOT}/status/source_files.sha256"
sha256sum -c "${RUN_ROOT}/status/source_files.sha256"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/preflight"

"${PYTHON}" -m unittest \
  tests.test_planned_corruption.PlannedCorruptionTests.test_safe_axis_groups_are_axis_pure_and_put_all_z_last \
  > "${RUN_ROOT}/status/focused_test.log" 2>&1
"${PYTHON}" -m py_compile "${EXECUTION_DIR}/evaluate_state_panels.py"

sinfo -h -p gpu -o '%P|%a|%D|%t|%G' > "${RUN_ROOT}/status/sinfo.txt"
squeue -h -u "${USER}" -o '%i|%P|%j|%T|%M|%R' > "${RUN_ROOT}/status/squeue.txt"
mkdir "${RUN_ROOT}/status/submission.lock"
JOB_ID="$(sbatch --parsable "${EXECUTION_DIR}/state_panels.sbatch")"
test -n "${JOB_ID}"

"${PYTHON}" - \
  "${RUN_ROOT}" \
  "${JOB_ID}" \
  "$(sha256sum "${RUN_ROOT}/source.tar" | cut -d' ' -f1)" \
  "$(sha256sum "${RUN_ROOT}/status/source_files.sha256" | cut -d' ' -f1)" <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
record = {
    "schema": "evidence_first_dlm_state_panel_submission_v1",
    "status": "complete",
    "identity": "h1_dlm_state_panels_b0_v1",
    "job_id": sys.argv[2],
    "partition": "gpu",
    "source_archive_sha256": sys.argv[3],
    "source_files_manifest_sha256": sys.argv[4],
    "checkpoint_arm": "B0",
    "training": False,
    "sun": False,
    "automatic_b3_submission": False,
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
}
(root / "status" / "submission_record.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
sha256sum "${RUN_ROOT}/status/submission_record.json" \
  > "${RUN_ROOT}/status/submission_record.sha256"
trap - EXIT
echo "state_panel_job=${JOB_ID}"

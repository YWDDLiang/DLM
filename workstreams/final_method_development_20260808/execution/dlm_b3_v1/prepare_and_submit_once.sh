#!/bin/bash
set -Eeuo pipefail
umask 077

if [ "$#" -ne 1 ]; then
  echo "usage: $0 SOURCE_ARCHIVE" >&2
  exit 2
fi

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_v1"
PANEL_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v1/panels"
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
test -f "${PANEL_ROOT}/state_panel_manifest.json"
test -f "${PANEL_ROOT}/terminal_report.json"
test -f "${PANEL_ROOT}/_SUCCESS"
mkdir -p "${RUN_ROOT}/source" "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
cp "${ARCHIVE}" "${RUN_ROOT}/source.tar"
tar -xf "${RUN_ROOT}/source.tar" -C "${RUN_ROOT}/source"

SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_b3_v1"
test -f "${EXECUTION_DIR}/SOURCE_FILES.txt"
cd "${SOURCE_ROOT}"
while IFS= read -r relative; do
  test -n "${relative}" || continue
  test -f "${relative}"
  sha256sum "${relative}"
done < "${EXECUTION_DIR}/SOURCE_FILES.txt" > "${RUN_ROOT}/status/source_files.sha256"
sha256sum -c "${RUN_ROOT}/status/source_files.sha256"
sha256sum "${PANEL_ROOT}/state_panel_manifest.json" \
  > "${RUN_ROOT}/status/frozen_panel_manifest.sha256"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/preflight"

"${PYTHON}" -m py_compile \
  "${SOURCE_ROOT}/scripts/llada_sft.py" \
  "${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/dlm_state_panels_v1/score_frozen_state_panels.py"
"${PYTHON}" - "${PANEL_ROOT}/terminal_report.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "complete"
    or report.get("checkpoint_arm") != "B0"
    or report.get("automatic_b3_submission") is not False
):
    raise SystemExit("B0 state-panel terminal is not eligible for B3 submission")
PY

sinfo -h -p gpu -o '%P|%a|%D|%t|%G' > "${RUN_ROOT}/status/sinfo.txt"
squeue -h -u "${USER}" -o '%i|%P|%j|%T|%M|%R' > "${RUN_ROOT}/status/squeue.txt"
mkdir "${RUN_ROOT}/status/submission.lock"
TRAIN_JOB="$(sbatch --parsable "${EXECUTION_DIR}/train_b3.sbatch")"
test -n "${TRAIN_JOB}"
SCORE_JOB="$(sbatch --parsable --dependency="afterany:${TRAIN_JOB}" "${EXECUTION_DIR}/score_b3_panels.sbatch")"
test -n "${SCORE_JOB}"

"${PYTHON}" - \
  "${RUN_ROOT}" \
  "${TRAIN_JOB}" \
  "${SCORE_JOB}" \
  "$(sha256sum "${RUN_ROOT}/source.tar" | cut -d' ' -f1)" \
  "$(sha256sum "${RUN_ROOT}/status/source_files.sha256" | cut -d' ' -f1)" \
  "$(cut -d' ' -f1 "${RUN_ROOT}/status/frozen_panel_manifest.sha256")" <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1])
record = {
    "schema": "evidence_first_dlm_b3_submission_v1",
    "status": "complete",
    "identity": "h1_dlm_b3_safe_axis_2to1_v1",
    "training_job_id": sys.argv[2],
    "state_panel_score_job_id": sys.argv[3],
    "score_dependency": f"afterany:{sys.argv[2]}",
    "training_partition": "gpu",
    "training_a800": 2,
    "source_archive_sha256": sys.argv[4],
    "source_files_manifest_sha256": sys.argv[5],
    "frozen_b0_state_panel_manifest_sha256": sys.argv[6],
    "initialization": "B0",
    "training_policy": "iid:d2_safe_axis=2:1",
    "optimizer_updates": 1696,
    "automatic_body64_submission": False,
    "automatic_ratio_sweep": False,
    "automatic_downstream": False,
    "automatic_sun": False,
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
}
(root / "status" / "submission_record.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
sha256sum "${RUN_ROOT}/status/submission_record.json" \
  > "${RUN_ROOT}/status/submission_record.sha256"
trap - EXIT
echo "b3_training_job=${TRAIN_JOB}"
echo "b3_state_panel_score_job=${SCORE_JOB}"
echo "automatic_body64_submission=false"

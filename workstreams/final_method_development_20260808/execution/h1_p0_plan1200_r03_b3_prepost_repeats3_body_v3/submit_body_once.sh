#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/body_source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
LOCK="$RUN_ROOT/status/body_submission.lock"
RECORD="$RUN_ROOT/status/body_submission_record.json"

test -f "$RUN_ROOT/status/body_preparation_SUCCESS"
test -f "$RUN_ROOT/status/body_preflight_report.json"
test -f "$RUN_ROOT/mp_cache/completion_SUCCESS"
test ! -e "$LOCK"
test ! -e "$RECORD"
test ! -e "$RUN_ROOT/terminal_report.json"
test ! -d "$RUN_ROOT/arms/R03"
test ! -d "$RUN_ROOT/arms/B3"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"
mkdir "$LOCK"

write_partial_failure() {
  local stage="$1"
  local rc="$2"
  local r03_job="${3:-}"
  local b3_job="${4:-}"
  "$PYTHON" - "$LOCK/partial_submission_failure.json" "$stage" "$rc" "$SOURCE_SHA" "$r03_job" "$b3_job" <<'PY'
import json
import os
import sys
from pathlib import Path

record = {
    "schema": "h1_plan1200_body_partial_submission_failure_v3",
    "status": "failed_closed",
    "failed_stage": sys.argv[2],
    "sbatch_return_code": int(sys.argv[3]),
    "source_manifest_sha256": sys.argv[4],
    "R03_array_job": sys.argv[5] or None,
    "B3_array_job": sys.argv[6] or None,
    "submitted_jobs_cancelled": False,
    "automatic_retry": False,
    "automatic_training": False,
    "automatic_rl": False,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

set +e
R03_JOB="$(
  cd "$PROJECT"
  sbatch --parsable \
    --job-name=h1-p1200-R03 \
    --export=ALL,H1_PLAN1200_BODY_SOURCE_SHA256="$SOURCE_SHA",H1_ARM=R03 \
    "$SOURCE/arm_pipeline.sbatch"
)"
R03_RC=$?
set -e
if [[ "$R03_RC" -ne 0 ]] || [[ ! "$R03_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure R03_array "$R03_RC" "$R03_JOB" ""
  [[ "$R03_RC" -ne 0 ]] || R03_RC=1
  exit "$R03_RC"
fi

set +e
B3_JOB="$(
  cd "$PROJECT"
  sbatch --parsable \
    --job-name=h1-p1200-B3 \
    --export=ALL,H1_PLAN1200_BODY_SOURCE_SHA256="$SOURCE_SHA",H1_ARM=B3 \
    "$SOURCE/arm_pipeline.sbatch"
)"
B3_RC=$?
set -e
if [[ "$B3_RC" -ne 0 ]] || [[ ! "$B3_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure B3_array "$B3_RC" "$R03_JOB" "$B3_JOB"
  [[ "$B3_RC" -ne 0 ]] || B3_RC=1
  exit "$B3_RC"
fi

set +e
ASSEMBLY_JOB="$(
  cd "$PROJECT"
  sbatch --parsable \
    --dependency="afterany:$R03_JOB:$B3_JOB" \
    --export=ALL,H1_PLAN1200_BODY_SOURCE_SHA256="$SOURCE_SHA" \
    "$SOURCE/assemble.sbatch"
)"
ASSEMBLY_RC=$?
set -e
if [[ "$ASSEMBLY_RC" -ne 0 ]] || [[ ! "$ASSEMBLY_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure assembly "$ASSEMBLY_RC" "$R03_JOB" "$B3_JOB"
  [[ "$ASSEMBLY_RC" -ne 0 ]] || ASSEMBLY_RC=1
  exit "$ASSEMBLY_RC"
fi

"$PYTHON" - "$RECORD" "$R03_JOB" "$B3_JOB" "$ASSEMBLY_JOB" "$SOURCE_SHA" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

record = {
    "schema": "h1_plan1200_body_submission_v3",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "separate_arm_arrays": True,
    "R03_array_job": int(sys.argv[2]),
    "B3_array_job": int(sys.argv[3]),
    "arm_arrays": "0-2%3",
    "arm_partition": "gpu",
    "assembly_job": int(sys.argv[4]),
    "assembly_partition": "normal",
    "assembly_dependency": f"afterany:{sys.argv[2]}:{sys.argv[3]}",
    "source_manifest_sha256": sys.argv[5],
    "automatic_retry": False,
    "automatic_training": False,
    "automatic_promotion": False,
    "automatic_rl": False,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
sha256sum "$RECORD" > "$RUN_ROOT/status/body_submission_record.sha256"
printf 'R03_array_job=%s\nB3_array_job=%s\nassembly_job=%s\n' "$R03_JOB" "$B3_JOB" "$ASSEMBLY_JOB"

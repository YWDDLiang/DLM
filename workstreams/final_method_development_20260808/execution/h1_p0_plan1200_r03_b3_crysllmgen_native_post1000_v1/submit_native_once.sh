#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/native1000_source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
LOCK="$RUN_ROOT/status/native1000_submission.lock"
RECORD="$RUN_ROOT/status/native1000_submission_record.json"

test -f "$RUN_ROOT/status/native1000_preparation_SUCCESS"
test -f "$RUN_ROOT/status/native1000_preflight_report.json"
test -f "$RUN_ROOT/status/body_submission_record.json"
test -f "$RUN_ROOT/native_mp_cache/completion_SUCCESS"
test ! -e "$LOCK"
test ! -e "$RECORD"
test ! -e "$RUN_ROOT/crysllmgen_native1000/terminal_report.json"
test ! -d "$RUN_ROOT/crysllmgen_native1000/arms"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"
readarray -t MAIN_JOBS < <("$PYTHON" - "$RUN_ROOT/status/body_submission_record.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d["R03_array_job"])
print(d["B3_array_job"])
print(d["assembly_job"])
PY
)
R03_MAIN="${MAIN_JOBS[0]}"
B3_MAIN="${MAIN_JOBS[1]}"
MAIN_ASSEMBLY="${MAIN_JOBS[2]}"
[[ "$R03_MAIN" =~ ^[0-9]+$ && "$B3_MAIN" =~ ^[0-9]+$ && "$MAIN_ASSEMBLY" =~ ^[0-9]+$ ]]
mkdir "$LOCK"

write_partial_failure() {
  local stage="$1"
  local rc="$2"
  local r03_job="${3:-}"
  local b3_job="${4:-}"
  "$PYTHON" - "$LOCK/partial_submission_failure.json" "$stage" "$rc" "$SOURCE_SHA" "$r03_job" "$b3_job" <<'PY'
import json, os, sys
from pathlib import Path
record = {
    "schema": "h1_plan1200_native1000_partial_submission_failure_v1",
    "status": "failed_closed",
    "failed_stage": sys.argv[2],
    "sbatch_return_code": int(sys.argv[3]),
    "source_manifest_sha256": sys.argv[4],
    "R03_native_array_job": sys.argv[5] or None,
    "B3_native_array_job": sys.argv[6] or None,
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
R03_JOB="$(cd "$PROJECT" && sbatch --parsable \
  --dependency="afterok:$R03_MAIN" \
  --job-name=h1-native1000-R03 \
  --export=ALL,H1_NATIVE1000_SOURCE_SHA256="$SOURCE_SHA",H1_ARM=R03 \
  "$SOURCE/native_arm_pipeline.sbatch")"
R03_RC=$?
set -e
if [[ "$R03_RC" -ne 0 ]] || [[ ! "$R03_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure R03_array "$R03_RC" "$R03_JOB" ""
  [[ "$R03_RC" -ne 0 ]] || R03_RC=1
  exit "$R03_RC"
fi

set +e
B3_JOB="$(cd "$PROJECT" && sbatch --parsable \
  --dependency="afterok:$B3_MAIN" \
  --job-name=h1-native1000-B3 \
  --export=ALL,H1_NATIVE1000_SOURCE_SHA256="$SOURCE_SHA",H1_ARM=B3 \
  "$SOURCE/native_arm_pipeline.sbatch")"
B3_RC=$?
set -e
if [[ "$B3_RC" -ne 0 ]] || [[ ! "$B3_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure B3_array "$B3_RC" "$R03_JOB" "$B3_JOB"
  [[ "$B3_RC" -ne 0 ]] || B3_RC=1
  exit "$B3_RC"
fi

set +e
ASSEMBLY_JOB="$(cd "$PROJECT" && sbatch --parsable \
  --dependency="afterany:$R03_JOB:$B3_JOB:$MAIN_ASSEMBLY" \
  --export=ALL,H1_NATIVE1000_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/native_assemble.sbatch")"
ASSEMBLY_RC=$?
set -e
if [[ "$ASSEMBLY_RC" -ne 0 ]] || [[ ! "$ASSEMBLY_JOB" =~ ^[0-9]+$ ]]; then
  write_partial_failure assembly "$ASSEMBLY_RC" "$R03_JOB" "$B3_JOB"
  [[ "$ASSEMBLY_RC" -ne 0 ]] || ASSEMBLY_RC=1
  exit "$ASSEMBLY_RC"
fi

"$PYTHON" - "$RECORD" "$R03_MAIN" "$B3_MAIN" "$MAIN_ASSEMBLY" "$R03_JOB" "$B3_JOB" "$ASSEMBLY_JOB" "$SOURCE_SHA" <<'PY'
import datetime, json, os, sys
from pathlib import Path
record = {
    "schema": "h1_plan1200_crysllmgen_native1000_submission_v1",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "main_R03_array_job": int(sys.argv[2]),
    "main_B3_array_job": int(sys.argv[3]),
    "main_assembly_job": int(sys.argv[4]),
    "R03_native_array_job": int(sys.argv[5]),
    "B3_native_array_job": int(sys.argv[6]),
    "native_arrays": "0-2%3",
    "native_partition": "gpu",
    "R03_dependency": f"afterok:{sys.argv[2]}",
    "B3_dependency": f"afterok:{sys.argv[3]}",
    "native_assembly_job": int(sys.argv[7]),
    "native_assembly_partition": "normal",
    "native_assembly_dependency": f"afterany:{sys.argv[5]}:{sys.argv[6]}:{sys.argv[4]}",
    "source_manifest_sha256": sys.argv[8],
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
print(json.dumps(record, sort_keys=True))
PY
sha256sum "$RECORD" > "$RUN_ROOT/status/native1000_submission_record.sha256"
printf 'R03_native_array_job=%s\nB3_native_array_job=%s\nnative_assembly_job=%s\n' \
  "$R03_JOB" "$B3_JOB" "$ASSEMBLY_JOB"

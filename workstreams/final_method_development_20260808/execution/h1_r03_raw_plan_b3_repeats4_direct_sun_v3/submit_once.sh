#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_adapter_sha_identity_repair_v3"
SOURCE="$RUN_ROOT/source"
RECORD="$RUN_ROOT/status/submission_record.json"

test -f "$RUN_ROOT/status/preparation_SUCCESS"
test -f "$RUN_ROOT/mp_cache/completion_SUCCESS"
test -f "$RUN_ROOT/mp_cache/completion_manifest.json"
test ! -e "$RECORD"
test ! -e "$RUN_ROOT/terminal_report.json"
for cell in M00 M10 M01 M11; do
  test ! -e "$RUN_ROOT/cells/$cell"
done

SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

REPEAT_JOB="$(
  cd "$PROJECT"
  sbatch --parsable \
    --export=ALL,H1_R03B3_SOURCE_SHA256="$SOURCE_SHA" \
    "$SOURCE/repeat_pipeline.sbatch"
)"
set +e
ASSEMBLY_JOB="$({
  cd "$PROJECT"
  sbatch --parsable \
    --dependency=afterany:"$REPEAT_JOB" \
    --export=ALL,H1_R03B3_SOURCE_SHA256="$SOURCE_SHA" \
    "$SOURCE/assemble.sbatch"
} 2>"$RUN_ROOT/status/assembly_submission.stderr")"
ASSEMBLY_RC=$?
set -e
if [[ "$ASSEMBLY_RC" -ne 0 ]]; then
  python3 - "$RUN_ROOT/status/partial_submission_failure.json" "$REPEAT_JOB" "$SOURCE_SHA" "$ASSEMBLY_RC" <<'PY'
import json
import os
import sys
from pathlib import Path

record = {
    "schema": "h1_r03_raw_plan_b3_partial_submission_failure_v1",
    "status": "failed_closed",
    "repeat_array_job": sys.argv[2],
    "assembly_job": None,
    "source_manifest_sha256": sys.argv[3],
    "assembly_sbatch_return_code": int(sys.argv[4]),
    "repeat_array_cancelled": False,
    "automatic_retry": False,
    "automatic_downstream": False,
    "automatic_rl": False,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  exit "$ASSEMBLY_RC"
fi

python3 - "$RECORD" "$REPEAT_JOB" "$ASSEMBLY_JOB" "$SOURCE_SHA" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

record = {
    "schema": "h1_r03_raw_plan_b3_repeats4_submission_record_v1",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repeat_array_job": sys.argv[2],
    "repeat_array": "0-3%2",
    "repeat_cell_slots": ["M00", "M10", "M01", "M11"],
    "repeat_partition": "gpu",
    "assembly_job": sys.argv[3],
    "assembly_partition": "normal",
    "assembly_dependency": f"afterany:{sys.argv[2]}",
    "source_manifest_sha256": sys.argv[4],
    "historical_control_rerun": False,
    "automatic_checkpoint_reselection": False,
    "automatic_training": False,
    "automatic_downstream": False,
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

sha256sum "$RECORD" > "$RUN_ROOT/status/submission_record.sha256"
printf '%s\n%s\n' "$REPEAT_JOB" "$ASSEMBLY_JOB"

#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_p0_plan1200_r03_b3_prepost_repeats3_v1"
SOURCE="$RUN_ROOT/planner_source"
EXECUTION="$SOURCE/workstreams/final_method_development_20260808/execution/h1_p0_plan1200_r03_b3_prepost_repeats3_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test -f "$RUN_ROOT/status/preparation_SUCCESS"
test ! -e "$RUN_ROOT/status/submission.lock"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
mkdir "$RUN_ROOT/status/submission.lock"

export H1_PLAN1200_SOURCE_SHA256="$SOURCE_SHA"
ARRAY_JOB_ID="$(sbatch --parsable "$EXECUTION/planner1200.sbatch")"
[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]
export H1_PLAN1200_ARRAY_JOB_ID="$ARRAY_JOB_ID"
ASSEMBLY_JOB_ID="$(sbatch --parsable --dependency="afterany:$ARRAY_JOB_ID" "$EXECUTION/planner_assembly.sbatch")"
[[ "$ASSEMBLY_JOB_ID" =~ ^[0-9]+$ ]]

"$PYTHON" - "$RUN_ROOT/status/submission.lock/submission.json" "$ARRAY_JOB_ID" "$ASSEMBLY_JOB_ID" "$SOURCE_SHA" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1_p0_plan1200_planner_submission_v1",
    "planner_array_job_id": int(sys.argv[2]),
    "planner_assembly_job_id": int(sys.argv[3]),
    "dependency": f"afterany:{sys.argv[2]}",
    "planner_source_manifest_sha256": sys.argv[4],
    "planner_array": "0-2%3",
    "repeat_seeds": [17029, 27183, 31415],
    "automatic_body_submission": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
printf 'planner_array_job_id=%s\nplanner_assembly_job_id=%s\n' "$ARRAY_JOB_ID" "$ASSEMBLY_JOB_ID"

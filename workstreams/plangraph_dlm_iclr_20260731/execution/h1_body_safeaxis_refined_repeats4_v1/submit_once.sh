#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUN="$ROOT/runs/20260802_h1_body_safeaxis_refined_repeats4_v1"
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test ! -e "$RUN"
mkdir -p "$RUN/logs" "$RUN/status"

ARRAY_JOB="$(sbatch --parsable \
  --export=ALL,R03E_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/repeat_pipeline.sbatch")"
ASSEMBLY_JOB="$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB" \
  --export=ALL,R03E_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/assemble.sbatch")"

python - "$RUN/status/submission_record.json" "$ARRAY_JOB" "$ASSEMBLY_JOB" "$SOURCE_SHA" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1_r03e_submission_record_v1",
    "status": "complete",
    "repeat_array_job": sys.argv[2],
    "assembly_job": sys.argv[3],
    "dependency": f"afterok:{sys.argv[2]}",
    "source_manifest_sha256": sys.argv[4],
    "repeat_array": "0-3%2",
    "packed_arms_per_repeat": 2,
    "automatic_downstream": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY

sha256sum "$RUN/status/submission_record.json"

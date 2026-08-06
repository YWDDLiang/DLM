#!/bin/bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_schedule32_v1"
RUN="$ROOT/runs/20260802_h1_body_schedule32_v1"

if [[ -e "$RUN/submission_record.json" ]]; then
  echo "H1 body schedule32 already has a submission record" >&2
  exit 2
fi

source_sha="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')"
mkdir -p "$RUN/logs"
job_id="$(
  cd "$ROOT"
  sbatch --parsable \
    --export=ALL,H1_BODY_SCHEDULE32_SOURCE_SHA256="$source_sha" \
    "$SOURCE/schedule32.sbatch"
)"

python3 - "$RUN/submission_record.json" "$job_id" "$source_sha" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
record = {
    "schema": "h1_body_schedule32_submission_record_v1",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "job_id": sys.argv[2],
    "source_manifest_sha256": sys.argv[3],
    "automatic_downstream": False,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

printf '%s\n' "$job_id"

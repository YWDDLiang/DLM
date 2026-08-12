#!/bin/bash
set -Eeuo pipefail
umask 077
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_once_v2"
RUN="$ROOT/runs/20260812_h1_r03_h1a2_archived_first256_once_v2"
if [[ -e "$RUN" ]]; then
  echo "immutable v2 run root already exists" >&2
  exit 2
fi
source_sha="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')"
mkdir -p "$RUN/logs" "$RUN/status"
on_failure() { code=$?; touch "$RUN/status/PREPARATION_FAILURE"; exit "$code"; }
trap on_failure ERR
python "$SOURCE/preflight_fast.py" \
  --config "$SOURCE/CONFIG.json" --source-dir "$SOURCE" \
  --source-manifest-sha256 "$source_sha" --run-root "$RUN" \
  --output "$RUN/status/preflight_report.json"
touch "$RUN/status/PREFLIGHT_SUCCESS"
job_id="$(cd "$ROOT" && sbatch --parsable \
  --export=ALL,ARCHIVED_ONCE_V2_SOURCE_SHA256="$source_sha" \
  "$SOURCE/archived_once_v2.sbatch")"
python3 - "$RUN/submission_record.json" "$job_id" "$source_sha" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
record = {
    "schema": "h1_r03_h1a2_archived_first256_once_submission_v2",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "job_id": sys.argv[2],
    "source_manifest_sha256": sys.argv[3],
    "slurm_jobs": 1,
    "gpus": 1,
    "cpus": 8,
    "attempts_per_arm": 256,
    "repeat": 0,
    "planner_resampled": False,
    "checkpoint_rehash_performed": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN/status/SUBMISSION_SUCCESS"
trap - ERR
printf '%s\n' "$job_id"

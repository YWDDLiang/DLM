#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
STAGED="$PROJECT/workstreams/final_method_development_20260808/execution/h1_sun_official_gga_u_skip_unknown_reeval_v2_skipunknown2"
RUN="$PROJECT/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2"
EVAL_PYTHON="/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python"
: "${H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256:?source manifest SHA is required}"
if [[ "$#" -ne 0 ]] || [[ -e "$RUN" ]]; then
  echo "usage changed or immutable run already exists" >&2
  exit 2
fi
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "registered evaluation interpreter is missing" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
(cd "$STAGED" && sha256sum -c SOURCE_SHA256.txt)
"$EVAL_PYTHON" "$STAGED/self_test.py" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256"
"$EVAL_PYTHON" "$STAGED/collect_inputs.py" \
  --config "$STAGED/CONFIG.json" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  --audit-only

mkdir "$RUN"
mkdir "$RUN/source" "$RUN/status" "$RUN/logs"
set -o noclobber
: > "$RUN/status/preparation.lock"
set +o noclobber
cp -a "$STAGED/." "$RUN/source/"
SOURCE="$RUN/source"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
"$EVAL_PYTHON" "$SOURCE/collect_inputs.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  --run-root "$RUN"
touch "$RUN/status/inputs_SUCCESS"

echo "STAGE adopt 2630 completed fresh official MP fragments; 80 approved Yb rows become hull_unknown"
"$EVAL_PYTHON" "$SOURCE/adopt_failed_spool.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  --run-root "$RUN"
test -f "$RUN/official_mp_cache/completion_SUCCESS"
touch "$RUN/status/preparation_SUCCESS"

set -o noclobber
: > "$RUN/status/submission.lock"
set +o noclobber
repeat_job="$(sbatch --parsable \
  --array=0-15%16 \
  --export=H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256="$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  "$SOURCE/reevaluate.sbatch")"
python - "$RUN/status/submission_partial.json" "$repeat_job" "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" <<'PY'
import datetime as dt
import json
import os
import sys

path, repeat_job, source_sha = sys.argv[1:]
record = {
    "schema": "h1_sun_official_gga_u_skip_unknown_submission_partial_v2",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "reevaluation_array_job": repeat_job,
    "reevaluation_array": "0-15%16",
    "assembly_job": None,
    "source_manifest_sha256": source_sha,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
assembly_job="$(sbatch --parsable \
  --dependency=afterany:"$repeat_job" \
  --export=H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256="$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  "$SOURCE/assemble.sbatch")"
python - "$RUN/status/submission_record.json" "$repeat_job" "$assembly_job" "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" <<'PY'
import datetime as dt
import json
import os
import sys

path, repeat_job, assembly_job, source_sha = sys.argv[1:]
record = {
    "schema": "h1_sun_official_gga_u_skip_unknown_submission_v2",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "reevaluation_array_job": repeat_job,
    "reevaluation_array": "0-15%16",
    "assembly_job": assembly_job,
    "assembly_dependency": f"afterany:{repeat_job}",
    "source_manifest_sha256": source_sha,
    "prepare_invocations": 1,
    "generation_or_relaxation_rerun": False,
    "new_mp_queries": 0,
    "fresh_official_spool_rows_adopted": 2630,
    "approved_hull_unknown_chemsys": 80,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
touch "$RUN/status/submission_SUCCESS"
printf 'REEVALUATION_JOB=%s\nASSEMBLY_JOB=%s\n' "$repeat_job" "$assembly_job"

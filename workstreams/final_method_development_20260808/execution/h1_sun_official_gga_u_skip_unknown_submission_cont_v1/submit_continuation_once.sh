#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2"
SOURCE="$RUN/source"
CONT="$PROJECT/workstreams/final_method_development_20260808/execution/h1_sun_official_gga_u_skip_unknown_submission_cont_v1_qos1"
RUN_SOURCE_SHA=a1883c9e820b7ca1ebd795180fd9f7ecd71bf26d971c20d41c239c5819fff5e5
: "${H1_SUN_CONT_SOURCE_SHA256:?continuation manifest SHA is required}"
if [[ "$#" -ne 0 ]]; then
  echo "continuation takes no arguments" >&2
  exit 2
fi

(cd "$CONT" && sha256sum -c SOURCE_SHA256.txt)
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = "$RUN_SOURCE_SHA"
test -f "$RUN/status/preparation_SUCCESS"
test -f "$RUN/official_mp_cache/completion_SUCCESS"
test -f "$RUN/status/submission.lock"
test ! -e "$RUN/status/submission_partial.json"
test ! -e "$RUN/status/submission_record.json"
test ! -e "$RUN/status/submission_SUCCESS"
test ! -e "$RUN/status/submission_continuation.lock"
test ! -e "$RUN/status/submission_continuation_record.json"
test ! -e "$RUN/status/submission_continuation_SUCCESS"
test ! -e "$RUN/cells"
if compgen -G "$RUN/status/cell_*" > /dev/null; then
  echo "cell status exists before continuation" >&2
  exit 2
fi
python - "$RUN/official_mp_cache/completion_manifest.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["query_status"] == "complete_with_explicit_hull_unknown"
assert value["resolved_query_count"] == 2550
assert value["unresolved_query_count"] == 80
assert value["new_mp_queries"] == 0
PY

set -o noclobber
: > "$RUN/status/submission_continuation.lock"
set +o noclobber
reevaluation_job="$(sbatch --parsable \
  --export=H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256="$RUN_SOURCE_SHA" \
  "$CONT/reevaluate_all.sbatch")"
python - "$RUN/status/submission_continuation_partial.json" "$reevaluation_job" "$H1_SUN_CONT_SOURCE_SHA256" <<'PY'
import datetime as dt
import json
import os
import sys

path, reevaluation_job, source_sha = sys.argv[1:]
record = {
    "schema": "h1_sun_skip_unknown_submission_continuation_partial_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "reevaluation_job": reevaluation_job,
    "execution": "one_normal_job_with_16_parallel_cell_processes",
    "assembly_job": None,
    "continuation_source_manifest_sha256": source_sha,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
assembly_job="$(sbatch --parsable \
  --dependency=afterany:"$reevaluation_job" \
  --export=H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256="$RUN_SOURCE_SHA" \
  "$SOURCE/assemble.sbatch")"
python - "$RUN/status/submission_continuation_record.json" "$reevaluation_job" "$assembly_job" "$H1_SUN_CONT_SOURCE_SHA256" <<'PY'
import datetime as dt
import json
import os
import sys

path, reevaluation_job, assembly_job, source_sha = sys.argv[1:]
record = {
    "schema": "h1_sun_skip_unknown_submission_continuation_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "initial_submission_outcome": "no_job_created_qos_max_submit_array_16",
    "reevaluation_job": reevaluation_job,
    "execution": "one_normal_job_with_16_parallel_cell_processes",
    "assembly_job": assembly_job,
    "assembly_dependency": f"afterany:{reevaluation_job}",
    "continuation_source_manifest_sha256": source_sha,
    "run_source_manifest_sha256": "a1883c9e820b7ca1ebd795180fd9f7ecd71bf26d971c20d41c239c5819fff5e5",
    "new_mp_queries": 0,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
touch "$RUN/status/submission_continuation_SUCCESS"
printf 'REEVALUATION_JOB=%s\nASSEMBLY_JOB=%s\n' "$reevaluation_job" "$assembly_job"


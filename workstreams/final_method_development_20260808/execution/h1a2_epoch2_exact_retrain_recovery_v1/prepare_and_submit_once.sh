#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
STAGED="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exact_retrain_recovery_v1"
RUN="$PROJECT/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_EXACT_RETRAIN_SOURCE_SHA256:?source manifest SHA is required}"

test "$#" -eq 0
test ! -e "$RUN"
test -f "$STAGED/SOURCE_SHA256.txt"
(cd "$STAGED" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$STAGED/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_EXACT_RETRAIN_SOURCE_SHA256"
bash -n "$STAGED/prepare_and_submit_once.sh" "$STAGED/train_epoch2.sbatch"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$STAGED/self_test.py" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_EXACT_RETRAIN_SOURCE_SHA256"
"$PYTHON" "$STAGED/preflight.py" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_EXACT_RETRAIN_SOURCE_SHA256" \
  --phase prepare

mkdir "$RUN"
mkdir "$RUN/source" "$RUN/status" "$RUN/logs" "$RUN/outputs"
set -o noclobber
: > "$RUN/status/preparation.lock"
set +o noclobber
cp -a "$STAGED/." "$RUN/source/"
SOURCE="$RUN/source"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
"$PYTHON" "$SOURCE/preflight.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_EXACT_RETRAIN_SOURCE_SHA256" \
  --phase prepared \
  --output "$RUN/status/preparation_preflight_report.json"
: > "$RUN/status/preparation_SUCCESS"

set -o noclobber
: > "$RUN/status/submission.lock"
set +o noclobber
job_id="$(sbatch --parsable \
  --export=H1_EXACT_RETRAIN_SOURCE_SHA256="$H1_EXACT_RETRAIN_SOURCE_SHA256" \
  "$SOURCE/train_epoch2.sbatch")"
"$PYTHON" - "$RUN/status/submission_record.json" "$job_id" "$H1_EXACT_RETRAIN_SOURCE_SHA256" <<'PY'
import datetime as dt, json, os, sys
path, job_id, source_sha = sys.argv[1:]
record = {
    "schema": "h1a2_epoch2_exact_retrain_submission_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "training_job": job_id,
    "partition": "gpu",
    "gpus": 1,
    "cpus": 8,
    "source_manifest_sha256": source_sha,
    "prepare_invocations": 1,
    "downstream_jobs_submitted": False,
    "materials_project_credentials_present": False,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY
: > "$RUN/status/submission_SUCCESS"
printf 'TRAINING_JOB=%s\n' "$job_id"

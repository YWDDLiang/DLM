#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
STAGED="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_refined256_current_sun_cache_replay_env_repair_v2"
RUN="$ROOT/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2"
: "${H1_R03_REPLAY_SOURCE_SHA256:?source manifest SHA is required}"
if [[ "$#" -ne 1 ]]; then
  echo "usage: prepare_and_submit_once.sh ONE_TIME_KEY_FILE" >&2
  exit 2
fi
KEY_FILE="$1"
if [[ -e "$RUN" ]] || [[ ! -f "$KEY_FILE" ]] \
  || [[ "$(stat -c '%a' "$KEY_FILE")" != 600 ]]; then
  echo "immutable run already exists or one-time key carrier is invalid" >&2
  exit 2
fi

mkdir -p "$RUN/source" "$RUN/status" "$RUN/logs"
set -o noclobber
: > "$RUN/status/preparation.lock"
set +o noclobber
cp -a "$STAGED/." "$RUN/source/"
SOURCE="$RUN/source"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
if [[ "$CONDA_DEFAULT_ENV" != diff_meets_diff ]] \
  || [[ "$CONDA_PREFIX" != /public/home/jiaosz/miniconda3/envs/diff_meets_diff ]]; then
  echo "registered MP completion environment changed" >&2
  exit 2
fi
python "$SOURCE/self_test.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03_REPLAY_SOURCE_SHA256"
touch "$RUN/status/preparation_SUCCESS"

echo "STAGE login-node MP completion: exactly 92 missing historical chemsys"
python "$SOURCE/complete_cache.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03_REPLAY_SOURCE_SHA256" \
  --run-root "$RUN" \
  --key-file "$KEY_FILE"
test ! -e "$KEY_FILE"
test -f "$RUN/mp_cache/completion_SUCCESS"

set -o noclobber
: > "$RUN/status/submission.lock"
set +o noclobber
repeat_job="$(sbatch --parsable \
  --export=ALL,H1_R03_REPLAY_SOURCE_SHA256="$H1_R03_REPLAY_SOURCE_SHA256" \
  "$SOURCE/repeat.sbatch")"
assembly_job="$(sbatch --parsable \
  --dependency=afterany:"$repeat_job" \
  --export=ALL,H1_R03_REPLAY_SOURCE_SHA256="$H1_R03_REPLAY_SOURCE_SHA256" \
  "$SOURCE/assemble.sbatch")"
python - "$RUN/status/submission_record.json" "$repeat_job" "$assembly_job" "$H1_R03_REPLAY_SOURCE_SHA256" <<'PY'
import datetime as dt, json, os, sys
path, repeat_job, assembly_job, source_sha = sys.argv[1:]
record = {
    "schema": "h1_r03_refined256_current_sun_cache_replay_submission_v2",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "repeat_array_job": repeat_job,
    "repeat_array": "0-3%4",
    "assembly_job": assembly_job,
    "assembly_dependency": f"afterany:{repeat_job}",
    "source_manifest_sha256": source_sha,
    "prepare_invocations": 1,
    "automatic_retry_or_requeue": False,
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
printf 'REPEAT_JOB=%s\nASSEMBLY_JOB=%s\n' "$repeat_job" "$assembly_job"

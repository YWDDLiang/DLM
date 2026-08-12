#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_official_ehull_completion_v1"
RUN="$ROOT/runs/20260813_h1_r03_h1a2_archived_first256_official_ehull_completion_v1"
KEY_FILE="${1:?one-time MP key carrier path is required}"
EXPECTED_SOURCE_SHA256="${2:?frozen source-manifest SHA256 is required}"

if [[ -e "$RUN" ]]; then
  echo "run root already exists; refusing duplicate execution" >&2
  exit 1
fi
if [[ ! -f "$KEY_FILE" ]]; then
  echo "one-time MP key carrier is absent" >&2
  exit 1
fi
cleanup_key() {
  rm -f -- "$KEY_FILE"
}
trap cleanup_key EXIT

mkdir -p "$RUN/status"
printf '%s\n' "$EXPECTED_SOURCE_SHA256" > "$RUN/status/expected_source_manifest_sha256"
cp -a "$SOURCE" "$RUN/source"
printf '{"schema":"h1_r03_h1a2_archived_first256_official_ehull_submission_v1","credential_serialized":false,"slurm_jobs":0}\n' > "$RUN/submission_record.json"
touch "$RUN/status/SUBMISSION_LOCK"

set +e
"$ROOT/../.venvs/mp_api_0_45_13_emmet0_85_1_py310_v4_system/bin/python" \
  "$RUN/source/complete_ehull.py" \
  --source-dir "$RUN/source" \
  --source-manifest-sha256 "$EXPECTED_SOURCE_SHA256" \
  --run-root "$RUN" \
  --key-file "$KEY_FILE"
code=$?
set -e
printf '%s\n' "$code" > "$RUN/status/exit_code"
if [[ "$code" -eq 0 ]]; then
  touch "$RUN/status/PIPELINE_SUCCESS" "$RUN/_SUCCESS"
else
  touch "$RUN/status/PIPELINE_FAILURE"
fi
exit "$code"

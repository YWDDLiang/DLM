#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
STAGED="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
RUN="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
UPSTREAM="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"
test "$#" -eq 0
test ! -e "$RUN"
test -f "$UPSTREAM/status/generation_assembly_SUCCESS"
test -f "$UPSTREAM/status/combined_all_SUCCESS"
test -f "$UPSTREAM/generation_terminal_report.json"
test -f "$UPSTREAM/official_sun_inputs/inputs_SUCCESS"
test -f "$STAGED/SOURCE_SHA256.txt"
(cd "$STAGED" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$STAGED/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
test "$(sha256sum "$PROJECT/workstreams/final_method_development_20260808/execution/h1_a2_r03_prepost_sun256_official_recovery_v1_terminal_contract_repair_v3/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 7c470d346ca374b8fd42d3c14e130e1c79257246847afa821248a4c5482f20c2

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
"$PYTHON" "$STAGED/self_test.py"
"$PYTHON" "$STAGED/preflight.py" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256"

mkdir -p "$RUN/source" "$RUN/status" "$RUN/logs" "$RUN/preliminary"
: > "$RUN/status/preparation.lock"
on_exit() {
  rc=$?
  if [[ "$rc" -ne 0 ]]; then touch "$RUN/status/preparation_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT
cp -a "$STAGED/." "$RUN/source/"
SOURCE="$RUN/source"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
"$PYTHON" "$SOURCE/preflight.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --output "$RUN/status/preflight_report.json"
"$PYTHON" "$SOURCE/prepare_prequery_inputs.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN"
touch "$RUN/status/preparation_SUCCESS"
chmod -R a-w "$SOURCE"
trap - EXIT
printf 'OFFICIAL_PREPARATION=READY\nPREQUERY_WANTED=1076\nSLURM_SUBMITTED=0\n'

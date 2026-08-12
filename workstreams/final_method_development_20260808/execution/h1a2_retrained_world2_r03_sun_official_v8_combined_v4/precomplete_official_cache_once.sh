#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
SOURCE="$RUN/source"
WORK="$RUN/prequery_workspace"
OFFICIAL_PYTHON=/public/home/jiaosz/ywliang/ai4s/.venvs/mp_api_0_45_13_emmet0_85_1_py310_v4_system/bin/python
AUDIT_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"
test "$#" -eq 1
KEY_FILE="$1"
cleanup_key() { if [[ -e "$KEY_FILE" ]]; then rm -f -- "$KEY_FILE"; fi; }
trap cleanup_key EXIT
case "$KEY_FILE" in "$RUN"/.mp_key_once.*) ;; *) echo "invalid key carrier path" >&2; exit 2;; esac
test -f "$KEY_FILE"
test "$(stat -c '%a' "$KEY_FILE")" = 600
test "$(stat -c '%s' "$KEY_FILE")" -ge 20
test "$(stat -c '%s' "$KEY_FILE")" -le 256
test -f "$RUN/status/preparation_SUCCESS"
test -f "$RUN/status/prequery_inputs_SUCCESS"
test -f "$WORK/status/preliminary_assembly_SUCCESS"
test ! -e "$RUN/status/precompleted_official_cache.lock"
test ! -e "$RUN/precompleted_official_mp_cache"
test ! -e "$WORK/official_mp_cache"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
: > "$RUN/status/precompleted_official_cache.lock"
on_exit() {
  rc=$?
  cleanup_key
  if [[ "$rc" -ne 0 ]]; then touch "$RUN/status/precompleted_official_cache_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
"$AUDIT_PYTHON" "$SOURCE/run_frozen_official.py" \
  --script audit_official_cache.py \
  --script-sha256 02c21066e60b49552b70c2adfa9a1b6186af1cdc014eb79717bc6ea327b07cf5 \
  -- \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$WORK"
test -f "$WORK/status/official_cache_audit_SUCCESS"
"$OFFICIAL_PYTHON" "$SOURCE/run_frozen_official.py" \
  --script complete_official_cache.py \
  --script-sha256 ff5e82de653e192a4bb27e6005adc5c02ff976f36ca57ad82f72c0a2398b5a85 \
  -- \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$WORK" \
  --key-file "$KEY_FILE"
test ! -e "$KEY_FILE"
test -f "$WORK/official_mp_cache/completion_SUCCESS"
mv "$WORK/official_mp_cache" "$RUN/precompleted_official_mp_cache"
touch "$RUN/status/precompleted_official_cache_SUCCESS"
trap - EXIT
printf 'OFFICIAL_CACHE_PRECOMPLETION=SUCCESS\nSLURM_SUBMITTED=0\n'

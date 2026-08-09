#!/bin/bash
set -Eeuo pipefail
umask 077

if [[ "$#" -ne 3 ]] || [[ ! "$2" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: prepare_and_submit_once.sh ARCHIVE EXPECTED_ARCHIVE_SHA256 PRIVATE_MP_KEY_FILE" >&2
  exit 2
fi

ARCHIVE="$(readlink -f "$1")"
EXPECTED_ARCHIVE_SHA256="$2"
KEY_FILE="$(readlink -f "$3")"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_evidence_first_four_cell_direct_sun_gcd_mp_cache_identity_repair_v4"
TMP_ROOT="$PROJECT/runs/.20260810_h1_evidence_first_four_cell_direct_sun_gcd_mp_cache_identity_repair_v4.preparing.$$"

cleanup() {
  rc=$?
  if [[ -n "${KEY_FILE:-}" ]] && [[ -f "$KEY_FILE" ]]; then
    rm -f -- "$KEY_FILE"
  fi
  if [[ "$rc" -ne 0 ]] && [[ -d "$TMP_ROOT" ]]; then
    mv "$TMP_ROOT" "${TMP_ROOT}.FAILED"
  fi
}
trap cleanup EXIT

test -f "$ARCHIVE"
test -f "$KEY_FILE"
test "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECTED_ARCHIVE_SHA256"
test ! -e "$RUN_ROOT"
test ! -e "$TMP_ROOT"
(cd "$SELF_DIR" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SELF_DIR/SOURCE_SHA256.txt" | cut -d' ' -f1)"

mkdir -p "$TMP_ROOT/source" "$TMP_ROOT/logs" "$TMP_ROOT/status" "$TMP_ROOT/cells"
cp -a "$SELF_DIR/." "$TMP_ROOT/source/"
(cd "$TMP_ROOT/source" && sha256sum -c SOURCE_SHA256.txt)
printf '%s  %s\n' "$EXPECTED_ARCHIVE_SHA256" "$ARCHIVE" > "$TMP_ROOT/status/source_archive.sha256"
printf '%s  source/SOURCE_SHA256.txt\n' "$SOURCE_SHA" > "$TMP_ROOT/status/source_manifest.sha256"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$TMP_ROOT/source"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

python "$TMP_ROOT/source/complete_planner_mp_cache.py" \
  --config "$TMP_ROOT/source/CONFIG.json" \
  --source-dir "$TMP_ROOT/source" \
  --source-manifest-sha256 "$SOURCE_SHA" \
  --prepared-root "$TMP_ROOT" \
  --key-file "$KEY_FILE" \
  >"$TMP_ROOT/logs/mp_cache_completion.out" \
  2>"$TMP_ROOT/logs/mp_cache_completion.err"
test ! -e "$KEY_FILE"
unset KEY_FILE

python "$TMP_ROOT/source/preflight.py" \
  --config "$TMP_ROOT/source/CONFIG.json" \
  --source-dir "$TMP_ROOT/source" \
  --source-manifest-sha256 "$SOURCE_SHA" \
  --prepared-root "$TMP_ROOT" \
  --output "$TMP_ROOT/status/preflight_report.json"
sha256sum "$TMP_ROOT/status/preflight_report.json" > "$TMP_ROOT/status/preflight_report.sha256"
touch "$TMP_ROOT/status/preparation_SUCCESS"

mv "$TMP_ROOT" "$RUN_ROOT"
trap - EXIT
bash "$RUN_ROOT/source/submit_once.sh"

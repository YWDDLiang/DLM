#!/bin/bash
set -Eeuo pipefail
umask 077

if [[ "$#" -ne 2 ]] || [[ ! "$2" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: prepare_and_submit_once.sh ARCHIVE EXPECTED_ARCHIVE_SHA256" >&2
  exit 2
fi

ARCHIVE="$(readlink -f "$1")"
EXPECTED_ARCHIVE_SHA256="$2"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_evidence_first_four_cell_direct_sun_lf_repair_v2"
TMP_ROOT="$PROJECT/runs/.20260810_h1_evidence_first_four_cell_direct_sun_lf_repair_v2.preparing.$$"

test -f "$ARCHIVE"
test "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECTED_ARCHIVE_SHA256"
test ! -e "$RUN_ROOT"
test ! -e "$TMP_ROOT"
(cd "$SELF_DIR" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SELF_DIR/SOURCE_SHA256.txt" | cut -d' ' -f1)"

cleanup() {
  rc=$?
  if [[ "$rc" -ne 0 ]] && [[ -d "$TMP_ROOT" ]]; then
    mv "$TMP_ROOT" "${TMP_ROOT}.FAILED"
  fi
}
trap cleanup EXIT

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

python "$TMP_ROOT/source/preflight.py" \
  --config "$TMP_ROOT/source/CONFIG.json" \
  --source-dir "$TMP_ROOT/source" \
  --source-manifest-sha256 "$SOURCE_SHA" \
  --output "$TMP_ROOT/status/preflight_report.json"
sha256sum "$TMP_ROOT/status/preflight_report.json" > "$TMP_ROOT/status/preflight_report.sha256"
touch "$TMP_ROOT/status/preparation_SUCCESS"

mv "$TMP_ROOT" "$RUN_ROOT"
trap - EXIT
bash "$RUN_ROOT/source/submit_once.sh"

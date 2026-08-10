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
RUN_ID=20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_adapter_sha_identity_repair_v3
RUN_ROOT="$PROJECT/runs/$RUN_ID"
TMP_ROOT="$PROJECT/runs/.$RUN_ID.preparing.$$"
V4_SOURCE="$PROJECT/runs/20260810_h1_evidence_first_four_cell_direct_sun_gcd_mp_cache_identity_repair_v4/source"
RAW512="$PROJECT/runs/20260729_h1a2c_jointchem_v1/arms/P0/plan512/raw_generations.jsonl"
R03F_CACHE="$PROJECT/runs/20260803_h1_body_safeaxis_refined_repeats4_mpcomplete_v1/common_mp_thermo_snapshot.jsonl"

cleanup() {
  rc=$?
  if [[ "$rc" -ne 0 ]] && [[ -d "$TMP_ROOT" ]]; then
    mv "$TMP_ROOT" "${TMP_ROOT}.FAILED"
  fi
}
trap cleanup EXIT

test -f "$ARCHIVE"
test "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECTED_ARCHIVE_SHA256"
test ! -e "$RUN_ROOT"
test ! -e "$TMP_ROOT"
(cd "$SELF_DIR" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SELF_DIR/SOURCE_SHA256.txt" | cut -d' ' -f1)"
test "$(sha256sum "$V4_SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = cb9391a919bea71a8a96c4fb576901101aed6a40069b89f27c16fc38ea9c518d
(cd "$V4_SOURCE" && sha256sum -c SOURCE_SHA256.txt)

mkdir -p \
  "$TMP_ROOT/source" \
  "$TMP_ROOT/inputs" \
  "$TMP_ROOT/mp_cache" \
  "$TMP_ROOT/logs" \
  "$TMP_ROOT/status" \
  "$TMP_ROOT/cells"
cp -a "$SELF_DIR/." "$TMP_ROOT/source/"
(cd "$TMP_ROOT/source" && sha256sum -c SOURCE_SHA256.txt)
printf '%s  %s\n' "$EXPECTED_ARCHIVE_SHA256" "$ARCHIVE" > "$TMP_ROOT/status/source_archive.sha256"
printf '%s  source/SOURCE_SHA256.txt\n' "$SOURCE_SHA" > "$TMP_ROOT/status/source_manifest.sha256"

test "$(sha256sum "$RAW512" | cut -d' ' -f1)" = bfaf2f9aa92ef4212d11bc71484ae6a60be13fd7239f107f08e419190afedb3e
test "$(wc -l < "$RAW512")" -eq 512
head -n 256 "$RAW512" > "$TMP_ROOT/inputs/r03_p0_raw_plan_first256.jsonl"
test "$(wc -l < "$TMP_ROOT/inputs/r03_p0_raw_plan_first256.jsonl")" -eq 256
test "$(sha256sum "$TMP_ROOT/inputs/r03_p0_raw_plan_first256.jsonl" | cut -d' ' -f1)" = c29857e33bd89e94e2257d3b752a0c15bbe6953ac1b7d9f11e575056c6114f79

test "$(sha256sum "$R03F_CACHE" | cut -d' ' -f1)" = 56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917
test "$(wc -l < "$R03F_CACHE")" -eq 227
cp "$R03F_CACHE" "$TMP_ROOT/mp_cache/r03f_common_mp_thermo_snapshot.jsonl"
test "$(sha256sum "$TMP_ROOT/mp_cache/r03f_common_mp_thermo_snapshot.jsonl" | cut -d' ' -f1)" = 56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917
printf '%s  %s\n' \
  56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917 \
  r03f_common_mp_thermo_snapshot.jsonl \
  > "$TMP_ROOT/mp_cache/r03f_common_mp_thermo_snapshot.sha256"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$V4_SOURCE:$TMP_ROOT/source"
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

for script in "$TMP_ROOT/source"/*.sh "$TMP_ROOT/source"/*.sbatch; do
  bash -n "$script"
done
python - "$TMP_ROOT/source" <<'PY'
import ast
import pathlib
import sys

for path in sorted(pathlib.Path(sys.argv[1]).glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

python - "$TMP_ROOT/mp_cache/completion_manifest.json" "$RUN_ID" "$SOURCE_SHA" <<'PY'
import json
import os
import sys
from pathlib import Path

record = {
    "schema": "h1_ef_fourcell_mp_cache_completion_manifest_v1",
    "status": "complete_all_missing_resolved",
    "run_id": sys.argv[2],
    "source_manifest_sha256": sys.argv[3],
    "cache_provenance": "byte-frozen R03F common snapshot; no new query",
    "wanted_chemsys_count": 227,
    "missing_chemsys_count": 107,
    "external_query_performed": False,
    "query_status_counts": {"reused_frozen_r03f": 107},
    "api_key_serialized": False,
    "sample_retry_or_replacement_used": False,
    "completed_mp_hull_cache": {
        "path": "mp_cache/r03f_common_mp_thermo_snapshot.jsonl",
        "rows": 227,
        "sha256": "56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917",
        "all_rows_populated": True,
    },
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$TMP_ROOT/mp_cache/completion_SUCCESS"

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

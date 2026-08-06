#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_evaluation_repair_v8"
RUN="$ROOT/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8"
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test -f "$SOURCE/AUTHORIZATION.json"
test -f "$SOURCE/EXECUTION_MANIFEST.json"
test -f "$SOURCE/SOURCE_SHA256.txt"
test ! -e "$RUN"
mkdir -p "$RUN/logs" "$RUN/status" "$RUN/arms"

ARRAY_JOB="$(sbatch --parsable \
  --export=ALL,V8_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/evaluate_arm.sbatch")"
ASSEMBLY_JOB="$(sbatch --parsable \
  --dependency=afterany:"$ARRAY_JOB" \
  --export=ALL,V8_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/assemble.sbatch")"

python - "$RUN/status/submission_record.json" "$SOURCE_SHA" \
  "$ARRAY_JOB" "$ASSEMBLY_JOB" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1a2_v3_poststop_sun256_evaluation_repair_submission_v1",
    "status": "complete",
    "source_manifest_sha256": sys.argv[2],
    "jobs": {
        "packed_sun_array": sys.argv[3],
        "afterany_assembly": sys.argv[4],
    },
    "dependencies": ["packed_sun_array", "afterany:assembly"],
    "arms": ["M00", "M10", "M01", "M11"],
    "attempts_per_arm": 256,
    "evaluation_only": True,
    "reuses_frozen_v7_generation_refine800_direct": True,
    "generation_or_refinement_rerun": False,
    "direct_metrics_rerun": False,
    "mp_api_enabled": False,
    "formal_g3": False,
    "automatic_promotion": False,
    "automatic_downstream": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY

printf 'ARRAY_JOB=%s\n' "$ARRAY_JOB"
printf 'ASSEMBLY_JOB=%s\n' "$ASSEMBLY_JOB"

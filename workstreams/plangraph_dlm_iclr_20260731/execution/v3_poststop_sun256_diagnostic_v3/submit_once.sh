#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_diagnostic_v3"
RUN="$ROOT/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v3"
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test -f "$SOURCE/EXECUTION_MANIFEST.json"
test -f "$SOURCE/SOURCE_SHA256.txt"
test -f "$SOURCE/RUNTIME_REQUIRED_SHA256.txt"
test ! -e "$RUN"
mkdir -p "$RUN/logs" "$RUN/status"

ARM_JOB="$(sbatch --parsable \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/arm_pipeline.sbatch")"
ASSEMBLY_JOB="$(sbatch --parsable \
  --dependency=afterany:"$ARM_JOB" \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/assemble.sbatch")"

python - "$RUN/status/submission_record.json" "$SOURCE_SHA" \
  "$ARM_JOB" "$ASSEMBLY_JOB" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1a2_v3_poststop_sun256_submission_v3",
    "status": "complete",
    "source_manifest_sha256": sys.argv[2],
    "jobs": {
        "end_to_end_arm_array": sys.argv[3],
        "assembly": sys.argv[4],
    },
    "dependencies": [
        "end_to_end_arm_array",
        "afterany:assembly",
    ],
    "factorial_arms": ["M00", "M10", "M01", "M11"],
    "attempts_per_arm": 256,
    "diffusion_refinement_required": True,
    "diffusion_reverse_steps": 800,
    "direct_metrics_and_sun_after_refinement_in_same_array_element": True,
    "formal_g3": False,
    "automatic_promotion": False,
    "automatic_downstream": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(record, sort_keys=True))
PY

printf 'ARM_JOB=%s\n' "$ARM_JOB"
printf 'ASSEMBLY_JOB=%s\n' "$ASSEMBLY_JOB"

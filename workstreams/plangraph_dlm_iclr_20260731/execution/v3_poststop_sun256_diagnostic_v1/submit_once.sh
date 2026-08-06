#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_diagnostic_v1"
RUN="$ROOT/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1"
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test -f "$SOURCE/EXECUTION_MANIFEST.json"
test -f "$SOURCE/SOURCE_SHA256.txt"
test -f "$SOURCE/RUNTIME_REQUIRED_SHA256.txt"
test ! -e "$RUN"
mkdir -p "$RUN/logs" "$RUN/status"

PREP_JOB="$(sbatch --parsable \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/prepare.sbatch")"
GEN_JOB="$(sbatch --parsable \
  --dependency=afterok:"$PREP_JOB" \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/generate_refine.sbatch")"
EVAL_JOB="$(sbatch --parsable \
  --dependency=afterok:"$GEN_JOB" \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/evaluate.sbatch")"
ASSEMBLY_JOB="$(sbatch --parsable \
  --dependency=afterany:"$EVAL_JOB" \
  --export=ALL,V3_SUN_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/assemble.sbatch")"

python - "$RUN/status/submission_record.json" "$SOURCE_SHA" \
  "$PREP_JOB" "$GEN_JOB" "$EVAL_JOB" "$ASSEMBLY_JOB" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1a2_v3_poststop_sun256_submission_v1",
    "status": "complete",
    "source_manifest_sha256": sys.argv[2],
    "jobs": {
        "prepare": sys.argv[3],
        "generation_and_diffusion_refinement_array": sys.argv[4],
        "sun_evaluation_array": sys.argv[5],
        "assembly": sys.argv[6],
    },
    "dependencies": [
        "prepare",
        "afterok:generation_and_diffusion_refinement_array",
        "afterok:sun_evaluation_array",
        "afterany:assembly",
    ],
    "factorial_arms": ["M00", "M10", "M01", "M11"],
    "attempts_per_arm": 256,
    "diffusion_refinement_required": True,
    "diffusion_reverse_steps": 800,
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

printf 'PREP_JOB=%s\n' "$PREP_JOB"
printf 'GEN_JOB=%s\n' "$GEN_JOB"
printf 'EVAL_JOB=%s\n' "$EVAL_JOB"
printf 'ASSEMBLY_JOB=%s\n' "$ASSEMBLY_JOB"


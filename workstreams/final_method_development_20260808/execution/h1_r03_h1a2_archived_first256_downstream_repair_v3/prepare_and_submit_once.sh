#!/bin/bash
set -Eeuo pipefail
umask 077
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_downstream_repair_v3"
RUN="$ROOT/runs/20260812_h1_r03_h1a2_archived_first256_downstream_repair_v3"
UPSTREAM="$ROOT/runs/20260812_h1_r03_h1a2_archived_first256_once_v2"
REF_SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUNTIME="$REF_SOURCE/runtime"
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY
if [[ -e "$RUN" ]]; then
  echo "immutable downstream repair v3 run root already exists" >&2
  exit 2
fi
source_sha="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')"
mkdir -p "$RUN/logs" "$RUN/status"
on_failure() { code=$?; touch "$RUN/status/PREPARATION_FAILURE"; exit "$code"; }
trap on_failure ERR

python "$SOURCE/preflight_downstream.py" \
  --config "$SOURCE/CONFIG.json" --source-dir "$SOURCE" \
  --source-manifest-sha256 "$source_sha" --run-root "$RUN" \
  --output "$RUN/status/preflight_report.json"
touch "$RUN/status/PREFLIGHT_SUCCESS"

install -m 600 "$UPSTREAM/status/body_identity_gate.json" "$RUN/status/body_identity_gate.json"
touch "$RUN/status/UPSTREAM_BODY_REUSED"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
observed_module="$(cd "$RUN" && PYTHONPATH="$RUNTIME:$REF_SOURCE" python -c 'from pathlib import Path; import scripts.refine_dlm_with_crysllmgen as m; print(Path(m.__file__).resolve())')"
expected_module="$(readlink -f "$RUNTIME/scripts/refine_dlm_with_crysllmgen.py")"
if [[ "$observed_module" != "$expected_module" ]]; then
  echo "refiner import smoke test resolved $observed_module instead of $expected_module" >&2
  exit 2
fi
printf '%s\n' "$observed_module" > "$RUN/status/refiner_import_smoke.txt"
touch "$RUN/status/REFINER_IMPORT_PREFLIGHT_SUCCESS"

job_id="$(cd "$ROOT" && sbatch --parsable \
  --export=ALL,ARCHIVED_DOWNSTREAM_V3_SOURCE_SHA256="$source_sha" \
  "$SOURCE/downstream_once.sbatch")"
python3 - "$RUN/submission_record.json" "$job_id" "$source_sha" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "h1_r03_h1a2_archived_first256_downstream_submission_v3",
    "status": "complete",
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "job_id": sys.argv[2],
    "source_manifest_sha256": sys.argv[3],
    "slurm_jobs": 1,
    "gpus": 1,
    "cpus": 8,
    "attempts_per_arm": 256,
    "repeat": 0,
    "body_generation_rerun": False,
    "body_reused_from_v2": True,
    "checkpoint_rehash_performed": False,
    "engineering_repair": "python_import_precedence_only",
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN/status/SUBMISSION_SUCCESS"
trap - ERR
printf '%s\n' "$job_id"

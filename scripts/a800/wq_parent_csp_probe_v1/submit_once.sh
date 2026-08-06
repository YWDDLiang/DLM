#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
JOB=scripts/a800/wq_parent_csp_probe_v1/probe.sbatch
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_bridge_v2_plan.json
RECORD="$RUN/notes/wq_parent_csp_same_proposal_probe_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_parent_csp_same_proposal_probe_v1"

: "${DIAGNOSTIC_PATCH_SHA256:?caller must export DIAGNOSTIC_PATCH_SHA256}"
if [[ ! "$DIAGNOSTIC_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid diagnostic patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${DIAGNOSTIC_PATCH_SHA256}.json"
test -f "$JOB"
test -f "$PLAN"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"
gpu_rows="$(
  printf '%s\n' "$existing_rows" |
    awk -F '|' 'NF >= 3 && $3 ~ /gpu/ { print }'
)"
if [ -n "$gpu_rows" ]; then
  echo "probe waits for zero preexisting user GPU jobs" >&2
  printf '%s\n' "$gpu_rows" >&2
  exit 3
fi

python - "$CLAIM" "$DIAGNOSTIC_PATCH_SHA256" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_parent_csp_probe_submission_claim_v1",
    "diagnostic_execution_patch_sha256": sys.argv[2],
    "pid": os.getpid(),
    "submit_once": True,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,DIAGNOSTIC_PATCH_SHA256=$DIAGNOSTIC_PATCH_SHA256 $JOB"
job_id="$(sbatch --parsable \
  --export="ALL,DIAGNOSTIC_PATCH_SHA256=$DIAGNOSTIC_PATCH_SHA256" \
  "$JOB")"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid job ID: $job_id" >&2
  exit 4
fi

python - "$RECORD" "$DIAGNOSTIC_PATCH_SHA256" "$job_id" "$command" \
  "$existing_rows" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_parent_csp_probe_submission_v1",
    "status": "complete",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "diagnostic_execution_patch_sha256": sys.argv[2],
    "job_id": sys.argv[3],
    "sbatch_command": sys.argv[4],
    "preexisting_queue_rows": [
        row for row in sys.argv[5].splitlines() if row
    ],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit_minutes": 45
    },
    "attempts": 4,
    "retry_or_replacement": False,
    "long_training_submitted": False
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "parent_csp_probe_job_id=$job_id"

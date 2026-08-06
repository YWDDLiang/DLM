#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
JOB=scripts/a800/wq_parent_csp_probe_sup27407_v1/probe.sbatch
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_probe_sup27407_v1.json
RECORD="$RUN/notes/wq_parent_csp_same_proposal_sup27407_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_parent_csp_same_proposal_sup27407_v1"
PRIOR_RECORD="$RUN/notes/wq_parent_csp_same_proposal_probe_v1_submission.json"
PRIOR_CLAIM="$PRIOR_RECORD.claim"
PRIOR_OUTPUT="$RUN/outputs/wq_parent_csp_same_proposal_probe_v1"
PRIOR_STDERR="$RUN/logs/wq-parent-csp-probe-v1-27407.err"
FAILURE_AUDIT=runs/remote_audit/20260724_wq_parent_csp_probe_job27407_failure_v1.json
AUTHORIZATION=runs/remote_audit/20260724_wq_parent_csp_probe_sup27407_v1_authorization.json

: "${DIAGNOSTIC_PATCH_SHA256:?caller must export DIAGNOSTIC_PATCH_SHA256}"
if [[ ! "$DIAGNOSTIC_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid diagnostic patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${DIAGNOSTIC_PATCH_SHA256}.json"
test -f "$JOB"
test -f "$PLAN"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = \
  dc9f170d3e7ac9d2585bec9750329e87a746ab5069d797f7a9ecd80b2f2ee36e
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = \
  3015a21aaaf86c607b0249b148ef2320b462dcef9f5dbe7e77a0c93a151d4f19
test "$(sha256sum "$PRIOR_RECORD" | awk '{print $1}')" = \
  4cbd6d4b0c2cd5075c6c7581dd76d1974da51afbd188276d047ae8a4c149eb95
test "$(sha256sum "$PRIOR_CLAIM" | awk '{print $1}')" = \
  cedbbf866a65efa9f896070c98bf3c4032118d8cf4f53ad986b70d11e5b0f9b9
test "$(sha256sum "$PRIOR_STDERR" | awk '{print $1}')" = \
  e02933db1381af2c9ad6d07939dd216db71413c7243be40b39ca69a7885b08fd
test -d "$PRIOR_OUTPUT"
if find "$PRIOR_OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "job 27407 unexpectedly produced scientific output" >&2
  exit 2
fi
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
  echo "superseding probe waits for zero preexisting user GPU jobs" >&2
  printf '%s\n' "$gpu_rows" >&2
  exit 3
fi

python - "$CLAIM" "$DIAGNOSTIC_PATCH_SHA256" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_parent_csp_probe_sup27407_submission_claim_v1",
    "diagnostic_execution_patch_sha256": sys.argv[2],
    "supersedes_job_id": "27407",
    "prior_scientific_attempts": 0,
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
    "schema": "wq_parent_csp_probe_sup27407_submission_v1",
    "status": "complete",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "diagnostic_execution_patch_sha256": sys.argv[2],
    "job_id": sys.argv[3],
    "sbatch_command": sys.argv[4],
    "preexisting_queue_rows": [
        row for row in sys.argv[5].splitlines() if row
    ],
    "supersedes_job_id": "27407",
    "supersession_authorized": True,
    "prior_scientific_attempts": 0,
    "authorized_fix": (
        "forward diagnostic_execution_patch_sha256 into GateALock.load"
    ),
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit_minutes": 45,
    },
    "attempts": 4,
    "scientific_attempt_retry_or_replacement": False,
    "long_training_submitted": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "parent_csp_sup27407_job_id=$job_id"

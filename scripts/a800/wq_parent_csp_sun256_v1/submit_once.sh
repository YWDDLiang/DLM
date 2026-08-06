#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_sun256_v1.json
PLAN_SHA256=51a6047aa8a5063852c2b2569dfd02953b1b1038b683266012262e9aed7401c6
AUTHORIZATION=runs/remote_audit/20260724_wq_parent_csp_sun256_v1/authorization_record.json
AUTHORIZATION_SHA256=1da13c5b5623c67314cd042e2204c9adaa3d3eafcf0e3919da263ca1e5f962db
P0_AUDIT=runs/remote_audit/20260724_wq_parent_csp_probe_sup27407_job27409_terminal_audit_v1.json
P0_AUDIT_SHA256=04749ac74f617567cdcb43f4f86ae8c91ee5048ec1cca82ec870c2936926c824
JOB=scripts/a800/wq_parent_csp_sun256_v1/pipeline.sbatch
RECORD="$RUN/notes/wq_parent_csp_sun256_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_parent_csp_sun256_v1"

: "${EVALUATION_PATCH_SHA256:?caller must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid evaluation patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$PLAN_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$P0_AUDIT" | awk '{print $1}')" = "$P0_AUDIT_SHA256"
test -f "$JOB"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"

mkdir -p "$RUN/notes"
python - "$CLAIM" "$EVALUATION_PATCH_SHA256" "$PLAN_SHA256" \
  "$AUTHORIZATION_SHA256" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_parent_csp_sun256_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "evaluation_execution_patch_sha256": sys.argv[2],
    "plan_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "submitted_by_slurm_user": os.environ.get("USER"),
    "submit_once": True,
    "attempts": 256,
    "attempt_retry_or_replacement_used": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,EVALUATION_PATCH_SHA256=$EVALUATION_PATCH_SHA256 $JOB"
set +e
output="$(
  sbatch --parsable \
    --export="ALL,EVALUATION_PATCH_SHA256=$EVALUATION_PATCH_SHA256" \
    "$JOB" 2>&1
)"
rc=$?
set -e
if [ "$rc" -eq 0 ] && [[ "$output" =~ ^[0-9]+$ ]]; then
  status=complete
  job_id="$output"
  failure_message=""
else
  status=submission_failed_no_retry
  job_id=""
  failure_message="rc=$rc output=$output"
fi

python - "$RECORD" "$status" "$job_id" "$failure_message" \
  "$EVALUATION_PATCH_SHA256" "$PLAN_SHA256" "$AUTHORIZATION_SHA256" \
  "$command" "$existing_rows" <<'PY'
import json, os, sys
from pathlib import Path

(
    path, status, job_id, failure, execution_patch, plan_sha,
    authorization_sha, command, existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wq_parent_csp_sun256_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "job_id": job_id or None,
    "sbatch_command": command,
    "evaluation_execution_patch_sha256": execution_patch,
    "plan_sha256": plan_sha,
    "authorization_record_sha256": authorization_sha,
    "preexisting_queue_rows": [
        row for row in existing_rows.splitlines() if row
    ],
    "queue_policy": (
        "record_only_do_not_block_or_modify_unrelated_user_jobs"
    ),
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 16,
        "memory_gib": 96,
        "time_limit": "18:00:00",
    },
    "pipeline": [
        "generate_256",
        "adapt_final_structure",
        "crysllmgen_direct_metrics",
        "strict_and_meta_SUN",
    ],
    "attempts": 256,
    "attempt_retry_or_replacement_used": False,
    "long_training_submitted": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [ "$status" != complete ]; then
  echo "$failure_message" >&2
  exit 4
fi
echo "wq_parent_csp_sun256_job_id=$job_id"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_sun256_eval_sup27410_v1.json
AUTHORIZATION=runs/remote_audit/20260724_wq_parent_csp_sun256_eval_sup27410_v1/authorization_record.json
FAILURE_AUDIT=runs/remote_audit/20260724_wq_parent_csp_sun256_job27410_terminal_failure_v1.json
JOB=scripts/a800/wq_parent_csp_sun256_eval_sup27410_v1/evaluate.sbatch
RECORD="$RUN/notes/wq_parent_csp_sun256_eval_sup27410_v1_submission.json"
CLAIM="$RECORD.claim"
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1/generation.jsonl"
OUTPUT="$RUN/outputs/wq_parent_csp_sun256_eval_sup27410_v1"
GENERATION_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
FAILURE_AUDIT_SHA256=2937e3a98ab761811023b4d20c18a051379ddd59f10b3ae54be8eb11273df78d

: "${EVALUATION_PATCH_SHA256:?caller must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' "invalid evaluation patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$GENERATION_SHA256"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test -f "$PLAN"
test -f "$AUTHORIZATION"
test -f "$JOB"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

job_cpus="$(
  awk -F= '/^#SBATCH --cpus-per-task=/{print $2}' "$JOB"
)"
job_gpus="$(
  awk -F: '/^#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:/{print $NF}' "$JOB"
)"
if [[ ! "$job_cpus" =~ ^[0-9]+$ ]] \
  || [[ ! "$job_gpus" =~ ^[0-9]+$ ]] \
  || (( job_gpus < 1 || job_cpus > 8 * job_gpus )) \
  || (( job_cpus != 8 || job_gpus != 1 )); then
  printf '%s\n' "CPU/A800 hard policy failed before claim" >&2
  exit 3
fi
if [[ "$(grep -c '^conda activate diff_meets_diff$' "$JOB")" -ne 1 ]] \
  || grep -q 'conda activate crysllm' "$JOB"; then
  printf '%s\n' "single diff_meets_diff environment policy failed before claim" >&2
  exit 4
fi

plan_sha="$(sha256sum "$PLAN" | awk '{print $1}')"
authorization_sha="$(sha256sum "$AUTHORIZATION" | awk '{print $1}')"
job_sha="$(sha256sum "$JOB" | awk '{print $1}')"
existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"

mkdir -p "$RUN/notes"
python - "$CLAIM" "$EVALUATION_PATCH_SHA256" "$plan_sha" \
  "$authorization_sha" "$job_sha" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_parent_csp_sun256_eval_sup27410_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "supersedes_failed_job_id": "27410",
    "evaluation_execution_patch_sha256": sys.argv[2],
    "plan_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "sbatch_sha256": sys.argv[5],
    "submitted_by_slurm_user": os.environ.get("USER"),
    "source_attempts": 256,
    "source_regenerated": False,
    "attempt_retry_or_replacement_used": False,
    "submit_once": True,
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
if [[ "$rc" -eq 0 && "$output" =~ ^[0-9]+$ ]]; then
  status=complete
  job_id="$output"
  failure_message=""
else
  status=submission_failed_no_retry
  job_id=""
  failure_message="rc=$rc output=$output"
fi

python - "$RECORD" "$status" "$job_id" "$failure_message" \
  "$EVALUATION_PATCH_SHA256" "$plan_sha" "$authorization_sha" \
  "$job_sha" "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path, status, job_id, failure, execution_patch, plan_sha,
    authorization_sha, sbatch_sha, command, existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wq_parent_csp_sun256_eval_sup27410_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "supersedes_failed_job_id": "27410",
    "job_id": job_id or None,
    "sbatch_command": command,
    "evaluation_execution_patch_sha256": execution_patch,
    "plan_sha256": plan_sha,
    "authorization_record_sha256": authorization_sha,
    "sbatch_sha256": sbatch_sha,
    "preexisting_queue_rows": [
        row for row in existing_rows.splitlines() if row
    ],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 96,
        "time_limit": "18:00:00"
    },
    "environment": "diff_meets_diff",
    "pipeline": [
        "environment_import_smoke",
        "crysllmgen_direct_metrics_on_preserved_generation",
        "strict_and_meta_SUN_on_preserved_generation"
    ],
    "source_attempts": 256,
    "source_regenerated": False,
    "attempt_retry_or_replacement_used": False,
    "training_submitted": False,
    "submitted_by_slurm_user": os.environ.get("USER")
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ "$status" != complete ]]; then
  printf '%s\n' "$failure_message" >&2
  exit 5
fi
printf 'wq_parent_csp_sun256_eval_sup27410_job_id=%s\n' "$job_id"

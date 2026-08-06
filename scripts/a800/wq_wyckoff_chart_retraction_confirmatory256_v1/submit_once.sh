#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_wyckoff_chart_retraction_confirmatory256_v1
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_chart_retraction_confirmatory256_v1.json
CONTRACT_SHA256=293c026d2f371b592a81e8e4d3982b4cb65ae3b0d90b82bf72a639caae24b77a
AUTHORIZATION=runs/remote_audit/20260726_wq_wyckoff_chart_retraction_confirmatory256_v1/submission_authorization_record.json
JOB=scripts/a800/wq_wyckoff_chart_retraction_confirmatory256_v1/pipeline.sbatch
RECORD="$RUN/notes/${IDENTITY}_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/$IDENTITY"

: "${WTB256_EXECUTION_PATCH_SHA256:?caller must export WTB256_EXECUTION_PATCH_SHA256}"
: "${WTB256_AUTHORIZATION_SHA256:?caller must export WTB256_AUTHORIZATION_SHA256}"
if [[ ! "$WTB256_EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB256_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid WTB-256 execution/authorization identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB256_EXECUTION_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = \
  "$WTB256_AUTHORIZATION_SHA256"
test -f "$JOB"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

python - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
def integer(flag):
    match = re.search(rf"^#SBATCH --{re.escape(flag)}=(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"missing integer SBATCH field {flag}")
    return int(match.group(1))
gpu = re.search(
    r"^#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:(\d+)\s*$",
    text,
    re.MULTILINE,
)
if gpu is None:
    raise SystemExit("WTB-256 exact A800 request is missing")
gpus = int(gpu.group(1))
cpus = integer("cpus-per-task")
if (
    gpus != 1
    or cpus != 8
    or cpus > 8 * gpus
    or "#SBATCH --partition=gpu" not in text
    or "#SBATCH --mem=96G" not in text
    or "#SBATCH --time=18:00:00" not in text
    or "#SBATCH --array" in text
):
    raise SystemExit("WTB-256 fail-closed resource gate rejected the job")
print("wtb256_preclaim_resource_gate=PASS")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"
same_identity="$(
  printf '%s\n' "$existing_rows" \
    | awk -F'|' -v name="wq-wtb256-confirm-v1" '$4 == name {print}'
)"
if [ -n "$same_identity" ]; then
  echo "same WTB-256 job identity already exists" >&2
  exit 3
fi

mkdir -p "$RUN/notes"
python - "$CLAIM" "$WTB256_EXECUTION_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$WTB256_AUTHORIZATION_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "wq_wyckoff_chart_retraction_confirmatory256_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_chart_retraction_confirmatory256_v1",
    "execution_patch_sha256": sys.argv[2],
    "contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "resources": {"partition": "gpu", "a800": 1, "cpus": 8, "memory_gib": 96},
    "attempts_per_arm": 256,
    "arms": ["R", "U", "T"],
    "retry_or_replacement_allowed": False,
    "automatic_training_authorized": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,WTB256_EXECUTION_PATCH_SHA256=$WTB256_EXECUTION_PATCH_SHA256,WTB256_AUTHORIZATION_SHA256=$WTB256_AUTHORIZATION_SHA256 $JOB"
set +e
output="$(
  sbatch --parsable \
    --export="ALL,WTB256_EXECUTION_PATCH_SHA256=$WTB256_EXECUTION_PATCH_SHA256,WTB256_AUTHORIZATION_SHA256=$WTB256_AUTHORIZATION_SHA256" \
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
  "$WTB256_EXECUTION_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$WTB256_AUTHORIZATION_SHA256" "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path, status, job_id, failure, patch_sha, contract_sha,
    authorization_sha, command, existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wq_wyckoff_chart_retraction_confirmatory256_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_chart_retraction_confirmatory256_v1",
    "job_id": job_id or None,
    "sbatch_command": command,
    "execution_patch_sha256": patch_sha,
    "contract_sha256": contract_sha,
    "authorization_record_sha256": authorization_sha,
    "preexisting_queue_rows": [
        row for row in existing_rows.splitlines() if row
    ],
    "queue_policy": "record_only_do_not_modify_unrelated_jobs",
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 96,
        "time_limit": "18:00:00",
    },
    "pipeline": [
        "freeze_wq_sources_256",
        "paired_R_U_T_generation",
        "crysllmgen_direct_metrics",
        "exact_R5C_A100_protocol_on_A800_SUN",
        "paired_summary_and_promotion_lock",
    ],
    "attempts_per_arm": 256,
    "retry_or_replacement_used": False,
    "training_submitted": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

if [ "$status" != complete ]; then
  echo "$failure_message" >&2
  exit 4
fi
echo "wq_wyckoff_chart_retraction_confirmatory256_job_id=$job_id"

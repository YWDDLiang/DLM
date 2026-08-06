#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_llm_charge_stop_paired64_v1
JOB=scripts/a800/wq_llm_charge_stop_paired64_v1/pilot.sbatch
JOB_NAME=wq-chemstop-p64-v1
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_llm_charge_stop_paired64_v1.json
CONTRACT_SHA256=fdcd6b2b2a897fd4cc912e4d5f0e3053626c2cc19a7f510d5a7c0beb6d060307
AUTHORIZATION_LABEL=user_wq_llm_charge_stop_paired64_v1_2026-07-27
PROTOCOL=configs/experiments/wyckoff_codiffusion/protocol_v4.yaml
GATE="$RUN/outputs/gate_a_lock_v8.json"
RECORD="$RUN/notes/${IDENTITY}_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/$IDENTITY"
EXACT_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${WQ_CHARGE_STOP_PATCH_SHA256:?caller must export WQ_CHARGE_STOP_PATCH_SHA256}"
if [[ ! "$WQ_CHARGE_STOP_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid WQ charge-stop patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -x "$EXACT_PYTHON"
test -f ".artifacts/source_sync/authorized_patch_${WQ_CHARGE_STOP_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test -f "$JOB"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

"$EXACT_PYTHON" - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

job = Path(sys.argv[1]).read_text(encoding="utf-8")
def one(pattern):
    values = re.findall(pattern, job, re.MULTILINE)
    if len(values) != 1:
        raise SystemExit(f"resource directive mismatch: {pattern}")
    return values[0]
cpus = int(one(r"^#SBATCH --cpus-per-task=(\d+)$"))
gres = one(r"^#SBATCH --gres=(\S+)$")
match = re.fullmatch(r"gpu:NVIDIAA800-SXM4-80GB:(\d+)", gres)
gpus = 0 if match is None else int(match.group(1))
if (
    one(r"^#SBATCH --partition=(\S+)$") != "gpu"
    or gpus != 1
    or cpus != 8
    or cpus > 8 * gpus
    or one(r"^#SBATCH --mem=(\S+)$") != "64G"
    or one(r"^#SBATCH --time=(\S+)$") != "03:00:00"
    or "#SBATCH --array" in job
):
    raise SystemExit("WQ charge-stop resource gate failed")
print("wq_charge_stop_preclaim_resource_gate=PASS")
PY

PYTHONPATH="$ROOT" "$EXACT_PYTHON" - \
  "$ROOT" "$GATE" "$PROTOCOL" "$WQ_CHARGE_STOP_PATCH_SHA256" \
  "$AUTHORIZATION_LABEL" <<'PY'
import sys
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.gate import (
    GateALock,
    PATCH_ALLOWED_AUTHORIZATIONS,
)
from scripts.a800.install_authorized_patch import ALLOWED_AUTHORIZATIONS

root = Path(sys.argv[1]).resolve()
authorization = sys.argv[5]
if set(ALLOWED_AUTHORIZATIONS) != set(PATCH_ALLOWED_AUTHORIZATIONS):
    raise SystemExit("installer/runtime authorization registries differ")
if authorization not in PATCH_ALLOWED_AUTHORIZATIONS:
    raise SystemExit("WQ charge-stop authorization is not registered")
loaded = GateALock.load(
    root / sys.argv[2],
    project_root=root,
    protocol_path=root / sys.argv[3],
    execution_patch_manifest_sha256=sys.argv[4],
)
patch = loaded.execution_patch
if (
    patch is None
    or patch.get("ok") is not True
    or patch.get("errors")
    or patch.get("changed")
    or patch.get("authorization") != authorization
    or patch.get("manifest_sha256") != sys.argv[4]
):
    raise SystemExit("WQ charge-stop installed patch failed Gate A")
print("wq_charge_stop_preclaim_gate_a=PASS")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING \
    -o '%i|%P|%b|%j|%T' | sort -u
)"
if printf '%s\n' "$existing_rows" |
  awk -F '|' -v name="$JOB_NAME" '$4 == name { found=1 } END { exit !found }'; then
  echo "same WQ charge-stop job already exists" >&2
  exit 3
fi

mkdir -p "$RUN/notes"
"$EXACT_PYTHON" - "$CLAIM" "$WQ_CHARGE_STOP_PATCH_SHA256" \
  "$CONTRACT_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "wq_llm_charge_stop_paired64_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_llm_charge_stop_paired64_v1",
    "execution_patch_sha256": sys.argv[2],
    "contract_sha256": sys.argv[3],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit": "03:00:00",
    },
    "attempts_per_arm": 64,
    "arms": ["baseline", "charge_stop"],
    "retry_or_replacement_allowed": False,
    "automatic_downstream_authorized": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,WQ_CHARGE_STOP_PATCH_SHA256=$WQ_CHARGE_STOP_PATCH_SHA256 $JOB"
set +e
output="$(
  sbatch --parsable \
    --export="ALL,WQ_CHARGE_STOP_PATCH_SHA256=$WQ_CHARGE_STOP_PATCH_SHA256" \
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

"$EXACT_PYTHON" - "$RECORD" "$status" "$job_id" "$failure_message" \
  "$WQ_CHARGE_STOP_PATCH_SHA256" "$CONTRACT_SHA256" "$command" \
  "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path, status, job_id, failure, patch_sha, contract_sha,
    command, existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wq_llm_charge_stop_paired64_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_llm_charge_stop_paired64_v1",
    "job_id": job_id or None,
    "sbatch_command": command,
    "execution_patch_sha256": patch_sha,
    "contract_sha256": contract_sha,
    "preexisting_queue_rows": [
        row for row in existing_rows.splitlines() if row
    ],
    "queue_policy": "record_only_do_not_modify_unrelated_jobs",
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit": "03:00:00",
    },
    "attempts_per_arm": 64,
    "arms": ["baseline", "charge_stop"],
    "retry_or_replacement_used": False,
    "training_submitted": False,
    "automatic_downstream_submitted": False,
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
echo "wq_charge_stop_job_id=$job_id"

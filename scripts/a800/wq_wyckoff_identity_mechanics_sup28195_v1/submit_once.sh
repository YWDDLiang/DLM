#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_wyckoff_identity_mechanics_sup28195_v1
JOB=scripts/a800/wq_wyckoff_identity_mechanics_sup28195_v1/mechanics.sbatch
JOB_NAME=wq-wtb-idv2-32
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_identity_mechanics_sup28195_v1.json
CONTRACT_SHA256=6ca0d3f292aff8fcaedd97566fc5b2367bc17c24f6a075b5a47490613cb5663d
AUTHORIZATION=diagnostics/authorization_records/wq_wyckoff_identity_mechanics_sup28195_v1.json
AUTHORIZATION_SHA256=d97e82caa5421a860a1943b50b17effabfd204980e994696041e692467cce417
AUTHORIZATION_LABEL=user_wq_wyckoff_identity_mechanics_sup28195_v1_2026-07-27
PROTOCOL=configs/experiments/wyckoff_codiffusion/protocol_v4.yaml
GATE="$RUN/outputs/gate_a_lock_v8.json"
SOURCE_ROOT="$RUN/outputs/wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/sources"
SOURCE="$SOURCE_ROOT/source_attempts.jsonl"
SOURCE_SHA256=3246a24d2595ae760e15f402222d6730a2a0fdbc404636254a7fa995559d56f2
SOURCE_REPORT="$SOURCE_ROOT/source_report.json"
SOURCE_REPORT_SHA256=9c6d5a9f2570f73b87ffa4c2bac898499588ea803cdf5803103408ce22323be9
JOB28195_AUDIT=runs/remote_audit/20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/terminal_failure_audit_job28195.json
JOB28195_AUDIT_SHA256=124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9
RECORD="$RUN/notes/${IDENTITY}_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/$IDENTITY"
EXACT_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${WTB_IDV2_PATCH_SHA256:?caller must export WTB_IDV2_PATCH_SHA256}"
: "${WTB_IDV2_AUTHORIZATION_SHA256:?caller must export WTB_IDV2_AUTHORIZATION_SHA256}"
if [[ ! "$WTB_IDV2_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB_IDV2_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$WTB_IDV2_AUTHORIZATION_SHA256" != "$AUTHORIZATION_SHA256" ]; then
  echo "invalid WTB identity-v2 patch/authorization identity" >&2
  exit 2
fi

cd "$ROOT"
test -x "$EXACT_PYTHON"
test -f ".artifacts/source_sync/authorized_patch_${WTB_IDV2_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | awk '{print $1}')" = \
  "$SOURCE_REPORT_SHA256"
test "$(sha256sum "$JOB28195_AUDIT" | awk '{print $1}')" = \
  "$JOB28195_AUDIT_SHA256"
test -f "$JOB"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

"$EXACT_PYTHON" - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

job = Path(sys.argv[1]).read_text(encoding="utf-8")

def one(pattern: str, name: str) -> str:
    values = re.findall(pattern, job, flags=re.MULTILINE)
    if len(values) != 1:
        raise SystemExit(f"expected exactly one {name} directive")
    return values[0]

partition = one(r"^#SBATCH --partition=(\S+)$", "partition")
cpus = int(one(r"^#SBATCH --cpus-per-task=(\d+)$", "CPU"))
gres = one(r"^#SBATCH --gres=(\S+)$", "GRES")
memory = one(r"^#SBATCH --mem=(\S+)$", "memory")
limit = one(r"^#SBATCH --time=(\S+)$", "time")
match = re.fullmatch(r"gpu:NVIDIAA800-SXM4-80GB:(\d+)", gres)
if match is None:
    raise SystemExit("WTB identity-v2 exact A800 GRES is missing")
gpus = int(match.group(1))
if (
    partition != "gpu"
    or gpus != 1
    or cpus != 8
    or cpus > 8 * gpus
    or memory != "64G"
    or limit != "01:00:00"
    or "#SBATCH --array" in job
):
    raise SystemExit("WTB identity-v2 resource envelope changed")
print("wtb_identity_v2_preclaim_resource_gate=PASS")
PY

PYTHONPATH="$ROOT" "$EXACT_PYTHON" - \
  "$ROOT" "$GATE" "$PROTOCOL" "$WTB_IDV2_PATCH_SHA256" \
  "$AUTHORIZATION_LABEL" <<'PY'
import sys
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.gate import (
    GateALock,
    PATCH_ALLOWED_AUTHORIZATIONS,
)
from scripts.a800.install_authorized_patch import ALLOWED_AUTHORIZATIONS

root = Path(sys.argv[1]).resolve()
gate = root / sys.argv[2]
protocol = root / sys.argv[3]
patch_sha = sys.argv[4]
authorization = sys.argv[5]
if set(ALLOWED_AUTHORIZATIONS) != set(PATCH_ALLOWED_AUTHORIZATIONS):
    raise SystemExit("installer/runtime Gate-A authorization registries differ")
if authorization not in PATCH_ALLOWED_AUTHORIZATIONS:
    raise SystemExit("WTB identity-v2 authorization is not registered")
loaded = GateALock.load(
    gate,
    project_root=root,
    protocol_path=protocol,
    execution_patch_manifest_sha256=patch_sha,
)
patch = loaded.execution_patch
if (
    patch is None
    or patch.get("ok") is not True
    or patch.get("errors")
    or patch.get("changed")
    or patch.get("authorization") != authorization
    or patch.get("manifest_sha256") != patch_sha
):
    raise SystemExit("WTB identity-v2 installed patch failed pre-claim Gate-A")
print("wtb_identity_v2_preclaim_gate_a_lock=PASS")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING \
    -o '%i|%P|%b|%j|%T' | sort -u
)"
if printf '%s\n' "$existing_rows" |
  awk -F '|' -v name="$JOB_NAME" '$4 == name { found=1 } END { exit !found }'; then
  echo "same WTB identity-v2 job already exists" >&2
  exit 3
fi

mkdir -p "$RUN/notes"
"$EXACT_PYTHON" - "$CLAIM" "$WTB_IDV2_PATCH_SHA256" \
  "$CONTRACT_SHA256" "$AUTHORIZATION_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "wq_wyckoff_identity_mechanics_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_identity_mechanics_sup28195_v1",
    "supersedes_failed_job_id": 28195,
    "execution_patch_sha256": sys.argv[2],
    "contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit": "01:00:00",
    },
    "source_identity_audit_attempts": 256,
    "mechanics_attempts_per_arm": 32,
    "arms": ["R", "U", "T"],
    "development_panel_reused": True,
    "confirmatory_evidence": False,
    "job28195_reinterpreted": False,
    "retry_or_replacement_allowed": False,
    "automatic_confirmatory_authorized": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,WTB_IDV2_PATCH_SHA256=$WTB_IDV2_PATCH_SHA256,WTB_IDV2_AUTHORIZATION_SHA256=$WTB_IDV2_AUTHORIZATION_SHA256 $JOB"
set +e
output="$(
  sbatch --parsable \
    --export="ALL,WTB_IDV2_PATCH_SHA256=$WTB_IDV2_PATCH_SHA256,WTB_IDV2_AUTHORIZATION_SHA256=$WTB_IDV2_AUTHORIZATION_SHA256" \
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
  "$WTB_IDV2_PATCH_SHA256" "$CONTRACT_SHA256" "$AUTHORIZATION_SHA256" \
  "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path,
    status,
    job_id,
    failure,
    patch_sha,
    contract_sha,
    authorization_sha,
    command,
    existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wq_wyckoff_identity_mechanics_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_identity_mechanics_sup28195_v1",
    "supersedes_failed_job_id": 28195,
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
        "memory_gib": 64,
        "time_limit": "01:00:00",
    },
    "pipeline": [
        "permutation_safe_source_identity_256",
        "development_R_U_T_mechanics_32",
        "terminal_mechanics_gate",
    ],
    "development_panel_reused": True,
    "confirmatory_evidence": False,
    "job28195_reinterpreted": False,
    "training_submitted": False,
    "new_generation_submitted": False,
    "crysllmgen_or_sun_submitted": False,
    "retry_or_replacement_used": False,
    "automatic_confirmatory_authorized": False,
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
echo "wq_wyckoff_identity_mechanics_job_id=$job_id"

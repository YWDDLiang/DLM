#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
PLAN=configs/experiments/wyckoff_codiffusion/wq_existing22_chgnet_sun_v1.json
AUTHORIZATION=runs/remote_audit/20260725_wq_existing22_chgnet_sun_v1/authorization_record.json
FORMAL_AUDIT=runs/remote_audit/20260725_wq_existing22_projection_survival_v2_order_alignment/terminal_audit.json
SOURCE="$RUN/outputs/wq_composition_existing22_survival_v2_order_alignment"
JOB=scripts/a800/wq_existing22_chgnet_sun_v1/evaluate.sbatch
RECORD="$RUN/notes/wq_existing22_chgnet_sun_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_existing22_chgnet_sun_v1"
AUTHORIZATION_SHA256=1d3e99578417ef192817ba3d0e3d0caa60b1013bb78e3b54936da68a4c1ca570
FORMAL_AUDIT_SHA256=59d08149cc10818c3d80ff5402bf0f4988de28a623d876371f6ecec683680c55
STRUCTURES_SHA256=47c4cf0b858bb846a5f9dc4df6dafa31b4ce4f20bdfc40be63d5226aac6e475e
METRICS_SHA256=360514adaf189db5ec0f6618cfae84068f8919df048535e17397c24c55fd4f69
REPORT_SHA256=03cd5f5e8c95391775e2b129296c309113c8e54aef29b3975eab7e82bd70f209
TERMINAL_SHA256=a9903b8e8012b5992117e4872735c5e5b46d67e009805e0dd7588c7b0ed4c7cc

: "${EVALUATION_PATCH_SHA256:?caller must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' "invalid evaluation patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$FORMAL_AUDIT" | awk '{print $1}')" = "$FORMAL_AUDIT_SHA256"
test "$(sha256sum "$SOURCE/structures.jsonl" | awk '{print $1}')" = "$STRUCTURES_SHA256"
test "$(sha256sum "$SOURCE/attempt_metrics.jsonl" | awk '{print $1}')" = "$METRICS_SHA256"
test "$(sha256sum "$SOURCE/report.json" | awk '{print $1}')" = "$REPORT_SHA256"
test "$(sha256sum "$SOURCE/terminal_acceptance.json" | awk '{print $1}')" = "$TERMINAL_SHA256"
test -f "$PLAN"
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
  || (( job_gpus != 1 || job_cpus != 8 || job_cpus > 8 * job_gpus )); then
  printf '%s\n' "CPU/A800 hard policy failed before claim" >&2
  exit 3
fi
if [[ "$(grep -c '^conda activate diff_meets_diff$' "$JOB")" -ne 1 ]] \
  || grep -q 'conda activate crysllm' "$JOB" \
  || ! grep -q '^unset MP_API_KEY$' "$JOB" \
  || ! grep -q '^unset PMG_MAPI_KEY$' "$JOB"; then
  printf '%s\n' "single offline diff_meets_diff policy failed before claim" >&2
  exit 4
fi
if ! python - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

job = Path(sys.argv[1])
pattern = re.compile(
    r"(?<![A-Za-z])train(?:ing)?(?![A-Za-z])"
    r"|fine[-_ ]?tune|optimizer|backward",
    re.IGNORECASE,
)
markers = []
for line in job.read_text(encoding="utf-8").splitlines():
    if not pattern.search(line):
        continue
    normalized = line.strip()
    if normalized.endswith("\\"):
        normalized = normalized[:-1].rstrip()
    markers.append(normalized)

# These are evaluator data/cache arguments, not model-training entrypoints.
# Exact equality keeps the preclaim check fail-closed for every other marker.
expected = [
    "--train-csv reference/crysllmgen/data/mp_20/train.csv",
    (
        "--training-index-cache "
        "reference/crysllmgen/data/mp_20/.cache/"
        "train.csv.3a814f7b7bf29b1a.training_index.pkl"
    ),
]
if markers != expected:
    print(
        "evaluation-only job contains an unexpected training marker: "
        f"{markers!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
  exit 5
fi

plan_sha="$(sha256sum "$PLAN" | awk '{print $1}')"
job_sha="$(sha256sum "$JOB" | awk '{print $1}')"
existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"

mkdir -p "$RUN/notes"
python - "$CLAIM" "$EVALUATION_PATCH_SHA256" "$plan_sha" "$job_sha" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wqcodiff_existing22_chgnet_sun_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "evaluation_execution_patch_sha256": sys.argv[2],
    "contract_sha256": sys.argv[3],
    "sbatch_sha256": sys.argv[4],
    "authorization_record_sha256": (
        "1d3e99578417ef192817ba3d0e3d0caa"
        "60b1013bb78e3b54936da68a4c1ca570"
    ),
    "formal_survival_gate_result": "FAIL_17_OF_22_BELOW_20_OF_22",
    "formal_survival_gate_rewritten": False,
    "continuation_identity": "user_accepted_exploratory_gate",
    "attempts": 22,
    "reconstructed_structures": 17,
    "failed_placeholders": 5,
    "resources": {"a800": 1, "cpus": 8},
    "submitted_by_slurm_user": os.environ.get("USER"),
    "new_generation": False,
    "training": False,
    "retry_or_replacement_used": False,
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
  "$EVALUATION_PATCH_SHA256" "$plan_sha" "$job_sha" "$command" \
  "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    path, status, job_id, failure, execution_patch, plan_sha,
    sbatch_sha, command, existing_rows,
) = sys.argv[1:]
payload = {
    "schema": "wqcodiff_existing22_chgnet_sun_submission_v1",
    "status": status,
    "failure_message": failure or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "job_id": job_id or None,
    "sbatch_command": command,
    "evaluation_execution_patch_sha256": execution_patch,
    "contract_sha256": plan_sha,
    "authorization_record_sha256": (
        "1d3e99578417ef192817ba3d0e3d0caa"
        "60b1013bb78e3b54936da68a4c1ca570"
    ),
    "sbatch_sha256": sbatch_sha,
    "preexisting_queue_rows": [
        row for row in existing_rows.splitlines() if row
    ],
    "queue_gated": False,
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 96,
        "time_limit": "04:00:00"
    },
    "environment": "diff_meets_diff",
    "pipeline": [
        "environment_smoke",
        "all22_attempt_adapter_with_5_frozen_failed_placeholders",
        "exact_chgnet_r5c_strict_and_meta_sun",
        "tri_state_terminal_acceptance"
    ],
    "attempts": 22,
    "new_generation": False,
    "geometry_repair_or_rescue": False,
    "training_submitted": False,
    "attempt_retry_or_replacement_used": False,
    "submitted_by_slurm_user": os.environ.get("USER")
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ "$status" != complete ]]; then
  printf '%s\n' "$failure_message" >&2
  exit 6
fi
printf 'wq_existing22_chgnet_sun_job_id=%s\n' "$job_id"

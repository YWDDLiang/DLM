#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
JOB=scripts/a800/wq_schedule_correct_bridge_parity_v1/preflight.sbatch
PLAN=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_execution_v1.json
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_v1.json
CONTRACT_SHA256=d4f18bf74a1814d7de6d7a4d4934c615857edef364a039f371723aa1763b4c6b
AUTHORIZATION=diagnostics/authorization_records/wq_schedule_correct_bridge_parity_v1_remote_execution.json
AUTHORIZATION_SHA256=ee83f4a2ee3ea71c08c8f642cab5fdc374accf8498de1eb08846ca5cc92f6cab
RECORD="$RUN/notes/wq_schedule_correct_bridge_parity_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_schedule_correct_bridge_parity_v1"
JOB_NAME=wq-bridge-parity-v1

: "${BRIDGE_PARITY_PATCH_SHA256:?caller must export BRIDGE_PARITY_PATCH_SHA256}"
if [[ ! "$BRIDGE_PARITY_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid bridge-parity patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${BRIDGE_PARITY_PATCH_SHA256}.json"
test -f "$JOB"
test -f "$PLAN"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

python - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")

def one(pattern: str, name: str) -> str:
    values = re.findall(pattern, text, flags=re.MULTILINE)
    if len(values) != 1:
        raise SystemExit(f"expected exactly one {name} directive")
    return values[0]

partition = one(r"^#SBATCH --partition=(\S+)$", "partition")
cpus = int(one(r"^#SBATCH --cpus-per-task=(\d+)$", "CPU"))
gres = one(r"^#SBATCH --gres=(\S+)$", "GRES")
memory = one(r"^#SBATCH --mem=(\S+)$", "memory")
limit = one(r"^#SBATCH --time=(\S+)$", "time")
gpu_match = re.fullmatch(r"gpu:NVIDIAA800-SXM4-80GB:(\d+)", gres)
if not gpu_match:
    raise SystemExit("preflight must request exact A800 GRES")
gpus = int(gpu_match.group(1))
if (
    partition != "gpu"
    or gpus != 1
    or cpus != 8
    or cpus > 8 * gpus
    or memory != "64G"
    or limit != "01:00:00"
):
    raise SystemExit("preflight resource envelope changed")
if "#SBATCH --array" in text:
    raise SystemExit("bridge parity must be one non-array job")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING \
    -o '%i|%P|%b|%j|%T' | sort -u
)"
if printf '%s\n' "$existing_rows" |
  awk -F '|' -v name="$JOB_NAME" '$4 == name { found=1 } END { exit !found }'; then
  echo "same bridge-parity job identity already exists" >&2
  exit 3
fi

python - "$CLAIM" "$BRIDGE_PARITY_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$AUTHORIZATION_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_schedule_correct_bridge_parity_submission_claim_v1",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "pid": os.getpid(),
    "submit_once": True,
    "evaluation_only": True,
    "training_submitted": False,
    "retry_or_replacement_allowed": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,BRIDGE_PARITY_PATCH_SHA256=$BRIDGE_PARITY_PATCH_SHA256 $JOB"
job_id="$(sbatch --parsable \
  --export="ALL,BRIDGE_PARITY_PATCH_SHA256=$BRIDGE_PARITY_PATCH_SHA256" \
  "$JOB")"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid job ID: $job_id" >&2
  exit 4
fi

python - "$RECORD" "$BRIDGE_PARITY_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$AUTHORIZATION_SHA256" "$job_id" "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_schedule_correct_bridge_parity_submission_v1",
    "status": "complete",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "job_id": sys.argv[5],
    "sbatch_command": sys.argv[6],
    "preexisting_queue_rows": [
        row for row in sys.argv[7].splitlines() if row
    ],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit_minutes": 60,
        "cpu_per_a800_gate": "cpus<=8*a800",
    },
    "matrix": {
        "timesteps": [100, 200, 400, 800],
        "attempts_per_timestep": 8,
        "total_cells": 32,
    },
    "evaluation_only": True,
    "training_submitted": False,
    "new_generation_submitted": False,
    "mlip_submitted": False,
    "retry_or_replacement_used": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "wq_schedule_correct_bridge_parity_job_id=$job_id"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
JOB=scripts/a800/wq_schedule_correct_bridge_parity_sup28054_v1/preflight.sbatch
PLAN=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_sup28054_execution_v1.json
PLAN_SHA256=55277e0e754b4319b9e5a8a4dd23a3443849e127d6a02a889da466ace8bec0a8
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_sup28054_v1.json
CONTRACT_SHA256=18472b1f40147fb7f70c304647ed164c11469f577898007e72dfc103fd31fb26
OLD_CONTRACT=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_v1.json
OLD_CONTRACT_SHA256=d4f18bf74a1814d7de6d7a4d4934c615857edef364a039f371723aa1763b4c6b
AUTHORIZATION=diagnostics/authorization_records/wq_schedule_correct_bridge_parity_sup28054_v1_remote_execution.json
AUTHORIZATION_SHA256=d29f1d52945f3cbb4b6ccfd393dc1dc06f28f4c960d2d787eabb3691b962a914
FAILURE_AUDIT=diagnostics/failure_audits/wq_schedule_correct_bridge_parity_job28054.json
FAILURE_AUDIT_SHA256=99b7b57b80c6d097adaa22bf3fa39d761573f4e76f6c74b89b55947081c95dca
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1/generation.jsonl"
SOURCE_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
SOURCE_SCHEMA=wqcodiff_generation_attempt_v1
CHECKPOINT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
CHECKPOINT_SHA256=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
OLD_RECORD="$RUN/notes/wq_schedule_correct_bridge_parity_v1_submission.json"
OLD_RECORD_SHA256=1ad6ed0839cc05347f44f8b14fa98b9464745022673481c092e4c744817b74ee
OLD_CLAIM="$OLD_RECORD.claim"
OLD_CLAIM_SHA256=b7bd96bbcefabdb3fa08a120e9f7c0c1b0c61f78eefb46f7f836034d57276c74
OLD_STDOUT="$RUN/logs/wq-bridge-parity-v1-28054.out"
OLD_STDOUT_SHA256=fe9645ffd1edbd54501239e45600f220e02ef967437a356334dbcf0af613484f
OLD_STDERR="$RUN/logs/wq-bridge-parity-v1-28054.err"
OLD_STDERR_SHA256=4e67596c9542594e405415fa7beb9665e023e059fddb87359d0b618ddf4a784c
OLD_GPU="$RUN/logs/wq-bridge-parity-v1-28054.gpu.csv"
OLD_GPU_SHA256=cf59e497613dd489de92dfe2c0d06a0ee9bb86ef2a8b54aeabe22672924cc00a
OLD_TERMINAL="$RUN/outputs/wq_schedule_correct_bridge_parity_v1/terminal_report.json"
OLD_TERMINAL_SHA256=207340080d8a9c565e7ac3c1ba7b7961030e70cf699df40d48aac9cfc7d97ae0
RECORD="$RUN/notes/wq_schedule_correct_bridge_parity_sup28054_v1_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1"
JOB_NAME=wq-bridge-sup28054-v1

: "${BRIDGE_PARITY_SUP28054_PATCH_SHA256:?caller must export BRIDGE_PARITY_SUP28054_PATCH_SHA256}"
if [[ ! "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid bridge-parity supersession patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${BRIDGE_PARITY_SUP28054_PATCH_SHA256}.json"
test -f "$JOB"
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$PLAN_SHA256"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$OLD_CONTRACT" | awk '{print $1}')" = "$OLD_CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$OLD_RECORD" | awk '{print $1}')" = "$OLD_RECORD_SHA256"
test "$(sha256sum "$OLD_CLAIM" | awk '{print $1}')" = "$OLD_CLAIM_SHA256"
test "$(sha256sum "$OLD_STDOUT" | awk '{print $1}')" = "$OLD_STDOUT_SHA256"
test "$(sha256sum "$OLD_STDERR" | awk '{print $1}')" = "$OLD_STDERR_SHA256"
test "$(sha256sum "$OLD_GPU" | awk '{print $1}')" = "$OLD_GPU_SHA256"
test "$(sha256sum "$OLD_TERMINAL" | awk '{print $1}')" = "$OLD_TERMINAL_SHA256"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

python - "$JOB" "$OLD_CONTRACT" "$CONTRACT" "$SOURCE" \
  "$SOURCE_SCHEMA" <<'PY'
import json
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
if "#SBATCH --array" in job:
    raise SystemExit("bridge parity supersession must be one non-array job")

old = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
new = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
for key in ("parent", "matrix", "bridge_semantics", "gates", "model_selection"):
    if old[key] != new[key]:
        raise SystemExit(f"unauthorized scientific change in {key}")
old_panel = dict(old["source_panel"])
new_panel = dict(new["source_panel"])
old_schema = old_panel.pop("required_schema")
new_schema = new_panel.pop("required_schema")
if old_panel != new_panel:
    raise SystemExit("unauthorized source-panel change")
if (
    old_schema != "wq_parent_csp_probe_attempt_v1"
    or new_schema != sys.argv[5]
):
    raise SystemExit("supersession schema delta is not exact")

rows = [
    json.loads(line)
    for line in Path(sys.argv[4]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 256 or {row.get("schema") for row in rows} != {sys.argv[5]}:
    raise SystemExit("source rows do not match the corrected exact schema")
if {row.get("status") for row in rows} != {"succeeded"}:
    raise SystemExit("source statuses are not exact")
if len({row.get("attempt_id") for row in rows}) != 256:
    raise SystemExit("source attempt IDs are not unique")
if not all(isinstance(row.get("proposal_state"), dict) for row in rows):
    raise SystemExit("source proposal_state is missing")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING \
    -o '%i|%P|%b|%j|%T' | sort -u
)"
if printf '%s\n' "$existing_rows" |
  awk -F '|' -v name="$JOB_NAME" '$4 == name { found=1 } END { exit !found }'; then
  echo "same bridge-parity supersession job identity already exists" >&2
  exit 3
fi

python - "$CLAIM" "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" \
  "$CONTRACT_SHA256" "$AUTHORIZATION_SHA256" "$FAILURE_AUDIT_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_schedule_correct_bridge_parity_sup28054_submission_claim_v1",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_schedule_correct_bridge_parity_sup28054_v1",
    "supersedes_failed_job_id": "28054",
    "superseded_job_scientific_trajectory_attempts": 0,
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "superseded_job_failure_audit_sha256": sys.argv[5],
    "pid": os.getpid(),
    "submit_once": True,
    "evaluation_only": True,
    "training_submitted": False,
    "scientific_delta": {
        "field": "source_panel.required_schema",
        "old": "wq_parent_csp_probe_attempt_v1",
        "new": "wqcodiff_generation_attempt_v1",
    },
    "retry_or_replacement_allowed": False,
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,BRIDGE_PARITY_SUP28054_PATCH_SHA256=$BRIDGE_PARITY_SUP28054_PATCH_SHA256 $JOB"
job_id="$(sbatch --parsable \
  --export="ALL,BRIDGE_PARITY_SUP28054_PATCH_SHA256=$BRIDGE_PARITY_SUP28054_PATCH_SHA256" \
  "$JOB")"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid job ID: $job_id" >&2
  exit 4
fi

python - "$RECORD" "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" \
  "$CONTRACT_SHA256" "$AUTHORIZATION_SHA256" "$FAILURE_AUDIT_SHA256" \
  "$job_id" "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "wq_schedule_correct_bridge_parity_sup28054_submission_v1",
    "status": "complete",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_schedule_correct_bridge_parity_sup28054_v1",
    "supersedes_failed_job_id": "28054",
    "superseded_job_failure_audit_sha256": sys.argv[5],
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "job_id": sys.argv[6],
    "sbatch_command": sys.argv[7],
    "preexisting_queue_rows": [
        row for row in sys.argv[8].splitlines() if row
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
    "scientific_delta": {
        "field": "source_panel.required_schema",
        "old": "wq_parent_csp_probe_attempt_v1",
        "new": "wqcodiff_generation_attempt_v1",
        "source_bytes_changed": False,
        "matrix_changed": False,
        "parent_checkpoint_changed": False,
        "acceptance_gates_changed": False,
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

echo "wq_schedule_correct_bridge_parity_sup28054_job_id=$job_id"

#!/bin/bash
#SBATCH --job-name=wq-bridge-sup28054-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_sup28054_v1.json
CONTRACT_SHA256=18472b1f40147fb7f70c304647ed164c11469f577898007e72dfc103fd31fb26
EXECUTION_PLAN=configs/experiments/wyckoff_codiffusion/wq_schedule_correct_bridge_parity_sup28054_execution_v1.json
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
OLD_TERMINAL="$RUN/outputs/wq_schedule_correct_bridge_parity_v1/terminal_report.json"
OLD_TERMINAL_SHA256=207340080d8a9c565e7ac3c1ba7b7961030e70cf699df40d48aac9cfc7d97ae0
OUTPUT="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1"

: "${BRIDGE_PARITY_SUP28054_PATCH_SHA256:?submission must export BRIDGE_PARITY_SUP28054_PATCH_SHA256}"
if [[ ! "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "bridge parity supersession identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${BRIDGE_PARITY_SUP28054_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test -f "$EXECUTION_PLAN"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$OLD_RECORD" | awk '{print $1}')" = "$OLD_RECORD_SHA256"
test "$(sha256sum "$OLD_CLAIM" | awk '{print $1}')" = "$OLD_CLAIM_SHA256"
test "$(sha256sum "$OLD_TERMINAL" | awk '{print $1}')" = "$OLD_TERMINAL_SHA256"
test ! -e "$OUTPUT"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

python - "$SOURCE" "$SOURCE_SHA256" "$SOURCE_SCHEMA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if hashlib.sha256(path.read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit("source hash changed")
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 256:
    raise SystemExit(f"expected 256 frozen source rows; observed {len(rows)}")
if {row.get("schema") for row in rows} != {sys.argv[3]}:
    raise SystemExit("frozen source schema differs from supersession contract")
if {row.get("status") for row in rows} != {"succeeded"}:
    raise SystemExit("frozen source status differs from supersession contract")
if len({row.get("attempt_id") for row in rows}) != 256:
    raise SystemExit("frozen source attempt IDs are not unique")
if not all(isinstance(row.get("proposal_state"), dict) for row in rows):
    raise SystemExit("frozen source proposal_state is missing")
print("bridge_parity_sup28054_source_schema=PASS")
print("bridge_parity_sup28054_source_rows=256")
PY

python - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"bridge parity requires exactly one visible CUDA device; "
        f"observed {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
if "A800" not in name:
    raise SystemExit(f"bridge parity requires A800; observed {name}")
print(f"bridge_parity_device={name}")
print(f"bridge_parity_torch={torch.__version__}")
print(f"bridge_parity_cuda={torch.version.cuda}")
PY

usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python scripts/a800/run_wq_schedule_correct_bridge_parity_v1.py \
  --contract "$CONTRACT" \
  --source-generation "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" \
  --device cuda

python - "$OUTPUT/terminal_report.json" "$CONTRACT_SHA256" \
  "$BRIDGE_PARITY_SUP28054_PATCH_SHA256" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
observed = report.get("observed", {})
checks = report.get("gate_checks", {})
if (
    report.get("schema") != "wq_schedule_correct_bridge_parity_terminal_v1"
    or report.get("ok") is not True
    or report.get("acceptance") != "PASS"
    or report.get("contract_sha256") != sys.argv[2]
    or report.get("execution_patch_sha256") != sys.argv[3]
    or observed.get("terminal_cells") != 32
    or observed.get("successful_positive_volume_outputs") != 32
    or observed.get("failed_cells") != 0
    or observed.get("first_reverse_step_invalid_lattice_count") != 0
    or observed.get("nonfinite_trajectory_count") != 0
    or not checks
    or not all(value is True for value in checks.values())
    or report.get("training_performed") is not False
    or report.get("new_generation_performed") is not False
    or report.get("mlip_used") is not False
    or report.get("external_api_used") is not False
):
    raise SystemExit("bridge parity supersession terminal acceptance audit failed")
print("wq_schedule_correct_bridge_parity_sup28054_acceptance=PASS")
print("wq_schedule_correct_bridge_parity_sup28054_cells=32")
print("wq_schedule_correct_bridge_training_performed=false")
PY

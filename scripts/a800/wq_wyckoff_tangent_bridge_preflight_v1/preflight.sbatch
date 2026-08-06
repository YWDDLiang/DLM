#!/bin/bash
#SBATCH --job-name=wq-tangent-wtb32-v1
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
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_tangent_bridge_preflight_v1.json
CONTRACT_SHA256=125a19b6eca74fca8c7637830263406568c48ff51a0878f44202af0c3fa136f2
RUNNER=scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py
RUNNER_SHA256=edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3
TANGENT=crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py
TANGENT_SHA256=175c98dcf36ced9b4f0fee6a30845ba78c9a275114fdca3c55392bf9dafa97e8
AUTHORIZATION=runs/remote_audit/20260726_wq_wyckoff_tangent_bridge_preflight_v1/submission_authorization_record.json
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1/generation.jsonl"
SOURCE_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
CHECKPOINT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
CHECKPOINT_SHA256=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
U_AUDIT=runs/remote_audit/20260726_wq_schedule_correct_bridge_parity_sup28054_v1/terminal_audit.json
U_AUDIT_SHA256=4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
U_SELECTION="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/selection_manifest.json"
U_SELECTION_SHA256=547c521c96e3375ffc0665d07c9104d605d14cef2e79c92159339eefb5065cc1
U_ATTEMPTS="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/attempts.jsonl"
U_ATTEMPTS_SHA256=888caaba6683d1b953bf6a7f5a42d9da3d2b432df7a3b1b810b525328b555e94
U_TERMINAL="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/terminal_report.json"
U_TERMINAL_SHA256=730dc0f005676f08d160b37006f1d2d9ababddef5e6ebeb0b200f129e062032e
OUTPUT="$RUN/outputs/wq_wyckoff_tangent_bridge_preflight_v1"

: "${WTB32_PATCH_SHA256:?submission must export WTB32_PATCH_SHA256}"
: "${WTB32_AUTHORIZATION_SHA256:?submission must export WTB32_AUTHORIZATION_SHA256}"
if [[ ! "$WTB32_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB32_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "WTB-32 identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB32_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$TANGENT" | awk '{print $1}')" = "$TANGENT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$WTB32_AUTHORIZATION_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$U_AUDIT" | awk '{print $1}')" = "$U_AUDIT_SHA256"
test "$(sha256sum "$U_SELECTION" | awk '{print $1}')" = "$U_SELECTION_SHA256"
test "$(sha256sum "$U_ATTEMPTS" | awk '{print $1}')" = "$U_ATTEMPTS_SHA256"
test "$(sha256sum "$U_TERMINAL" | awk '{print $1}')" = "$U_TERMINAL_SHA256"
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

python - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit("WTB-32 requires exactly one visible CUDA device")
name = torch.cuda.get_device_name(0)
if "A800" not in name:
    raise SystemExit(f"WTB-32 requires A800; observed {name}")
print(f"wtb32_device={name}")
print(f"wtb32_torch={torch.__version__}")
print(f"wtb32_cuda={torch.version.cuda}")
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

python "$RUNNER" \
  --contract "$CONTRACT" \
  --source-generation "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --u-terminal-audit "$U_AUDIT" \
  --u-selection-manifest "$U_SELECTION" \
  --u-attempts "$U_ATTEMPTS" \
  --u-terminal-report "$U_TERMINAL" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$WTB32_PATCH_SHA256" \
  --device cuda

python - "$OUTPUT/terminal_report.json" "$CONTRACT_SHA256" \
  "$WTB32_PATCH_SHA256" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arms = report.get("arm_summaries", {})
checks = report.get("gate_checks", {})
if (
    report.get("schema") != "wq_wyckoff_tangent_preflight_terminal_v1"
    or report.get("ok") is not True
    or report.get("acceptance") != "PASS"
    or report.get("contract_sha256") != sys.argv[2]
    or report.get("execution_patch_sha256") != sys.argv[3]
    or set(arms) != {"F", "T"}
    or any(arms[arm].get("terminal_cells") != 32 for arm in ("F", "T"))
    or any(arms[arm].get("succeeded_cells") != 32 for arm in ("F", "T"))
    or not checks
    or not all(value is True for value in checks.values())
    or report.get("training_performed") is not False
    or report.get("new_generation_performed") is not False
    or report.get("u_rerun") is not False
    or report.get("mlip_used") is not False
    or report.get("external_api_used") is not False
    or report.get("retry_or_replacement_used") is not False
):
    raise SystemExit("WTB-32 terminal acceptance audit failed")
print("wq_wyckoff_tangent_bridge_preflight_acceptance=PASS")
print("wq_wyckoff_tangent_bridge_preflight_F_cells=32")
print("wq_wyckoff_tangent_bridge_preflight_T_cells=32")
PY

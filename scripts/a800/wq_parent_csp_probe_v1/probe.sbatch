#!/bin/bash
#SBATCH --job-name=wq-parent-csp-probe-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
ADAPTER="$TRAIN/epoch_03/adapter_final"
TRAINING_REPORT="$TRAIN/epoch_03/training_report.json"
OUTPUT="$RUN/outputs/wq_parent_csp_same_proposal_probe_v1"
ADAPTER_TRAINING_PATCH_SHA256=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491

: "${DIAGNOSTIC_PATCH_SHA256:?submission must export DIAGNOSTIC_PATCH_SHA256}"
if [[ ! "$DIAGNOSTIC_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "parent-CSP probe identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${DIAGNOSTIC_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH_SHA256}.json"
test "$(sha256sum "$TRAINING_REPORT" | awk '{print $1}')" = \
  1d4bb3275dd5f1a20b9743161b59c633faad6f008ebbf7881ad07ddd2c6dfaf9
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = \
  9e12192fbd9fd42a1cdc236a50a3e70f8f97e4b44353cb37986c72dafd21b45c
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
if torch.cuda.device_count() != 1 or "A800" not in torch.cuda.get_device_name(0):
    raise SystemExit("parent-CSP probe requires exactly one A800")
print(f"device={torch.cuda.get_device_name(0)}")
PY

mkdir "$OUTPUT"
usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python scripts/a800/probe_wq_proposal_with_parent_csp.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --csp-checkpoint /public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --llama-adapter "$ADAPTER" \
  --output-jsonl "$OUTPUT/generation.jsonl" \
  --attempt-ledger "$OUTPUT/attempt_ledger.jsonl" \
  --report "$OUTPUT/probe_report.json" \
  --experiment-id wq-parent-csp-same-proposal-probe-v1 \
  --pairing-id wq-epoch-selection-paired-noise-sup27104-v2 \
  --training-seed 11 \
  --sampling-seed 101 \
  --attempts 4 \
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --diagnostic-execution-patch-sha256 "$DIAGNOSTIC_PATCH_SHA256"

python - "$OUTPUT/probe_report.json" "$DIAGNOSTIC_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("schema") != "wq_parent_csp_probe_report_v1"
    or report.get("ok") is not True
    or report.get("terminal") != 4
    or report.get("submitted") != 4
    or report.get("succeeded", 0) + report.get("failed", 0) != 4
    or report.get("pairing_id")
    != "wq-epoch-selection-paired-noise-sup27104-v2"
    or report.get("retry_or_replacement_used") is not False
    or report.get("diagnostic_execution_patch_sha256") != sys.argv[2]
):
    raise SystemExit("parent-CSP probe execution audit failed")
print("parent_csp_probe_execution_acceptance=PASS")
if report["succeeded"] > 0:
    print("parent_csp_probe_scientific_gate=PASS")
else:
    print("parent_csp_probe_scientific_gate=FAIL")
PY

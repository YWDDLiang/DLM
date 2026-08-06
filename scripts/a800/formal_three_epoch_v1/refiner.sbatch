#!/bin/bash
#SBATCH --job-name=wq-ref-sup26679-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
OUTPUT="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26679_v1"
REPLACEMENT_OF_JOB_ID=26679

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]; then
  echo "formal refiner identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test ! -e "$OUTPUT"
mkdir -p "$RUN/outputs"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

python - <<'PY'
import torch
if torch.cuda.device_count() != 1 or "A800" not in torch.cuda.get_device_name(0):
    raise SystemExit("formal refiner requires one A800")
print(f"device={torch.cuda.get_device_name(0)}")
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

python scripts/a800/train_crysllmgen_wq_refiner.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --replacement-of-job-id "$REPLACEMENT_OF_JOB_ID" \
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
  --csp-checkpoint /public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt \
  --dataset data/wqcodiff/p1_v3/train.part-000-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-001-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-002-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-003-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-004-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-005-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-006-of-008.jsonl \
  --dataset data/wqcodiff/p1_v3/train.part-007-of-008.jsonl \
  --training-seed 11 \
  --output-dir "$OUTPUT" \
  --microbatch 128 \
  --run-role main \
  --prefetch-workers 15 \
  --prefetch-depth 30

python - "$OUTPUT/training_report.json" "$EXECUTION_PATCH_SHA256" <<'PY'
import json, math, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("ok") is not True
    or report.get("run_role") != "main"
    or report.get("training_seed") != 11
    or report.get("updates") != 100000
    or report.get("paper_eligible") is not True
    or report.get("execution_patch_sha256") != sys.argv[2]
    or report.get("replacement_of_job_id") != "26679"
    or report.get("prefetch_workers") != 15
    or report.get("prefetch_depth") != 30
    or not math.isfinite(float(report.get("walltime_s", float("nan"))))
):
    raise SystemExit("formal refiner acceptance failed")
checkpoint = Path(report["checkpoint"])
if not checkpoint.is_file():
    raise SystemExit("formal refiner checkpoint is missing")
print("formal_refiner_acceptance=PASS")
PY

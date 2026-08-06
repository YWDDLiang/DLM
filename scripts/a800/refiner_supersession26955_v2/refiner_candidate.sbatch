#!/bin/bash
#SBATCH --job-name=wq-ref-candidate-v2
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
PREFETCH_WORKERS=7
PREFETCH_DEPTH=14
REPLACEMENT_OF_JOB_ID=26955

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
: "${WQ_REFINER_MODE:?submission must export WQ_REFINER_MODE}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]; then
  echo "refiner supersession identity/resource contract changed" >&2
  exit 2
fi

case "$WQ_REFINER_MODE" in
  smoke)
    [ "${SLURM_JOB_NAME:-}" = wq-ref-smoke-p7-v2 ]
    OUTPUT="$RUN/outputs/refiner_prefetch7_stability_smoke_v2"
    RUN_ROLE=smoke
    EXPECTED_UPDATES=100
    ;;
  main)
    [ "${SLURM_JOB_NAME:-}" = wq-ref-sup26955-v2 ]
    OUTPUT="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26955_v2"
    RUN_ROLE=main
    EXPECTED_UPDATES=100000
    ;;
  *)
    echo "unknown WQ_REFINER_MODE=$WQ_REFINER_MODE" >&2
    exit 2
    ;;
esac

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test -f "$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1/training_report.json"
test ! -e "$OUTPUT"
mkdir -p "$RUN/outputs" "$RUN/logs"

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
    raise SystemExit("refiner supersession requires one A800")
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

args=(
  python scripts/a800/train_crysllmgen_wq_refiner.py
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml
  --source-bundle-sha256 "$BASE_SOURCE"
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256"
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream
  --csp-checkpoint /public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
  --dataset data/wqcodiff/p1_v3/train.part-000-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-001-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-002-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-003-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-004-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-005-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-006-of-008.jsonl
  --dataset data/wqcodiff/p1_v3/train.part-007-of-008.jsonl
  --training-seed 11
  --output-dir "$OUTPUT"
  --microbatch 128
  --run-role "$RUN_ROLE"
  --prefetch-workers "$PREFETCH_WORKERS"
  --prefetch-depth "$PREFETCH_DEPTH"
)
if [ "$WQ_REFINER_MODE" = smoke ]; then
  args+=(--smoke-updates 100)
else
  args+=(
    --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json"
    --replacement-of-job-id "$REPLACEMENT_OF_JOB_ID"
  )
fi
"${args[@]}"

python - "$OUTPUT/training_report.json" "$EXECUTION_PATCH_SHA256" \
  "$WQ_REFINER_MODE" "$EXPECTED_UPDATES" "$REPLACEMENT_OF_JOB_ID" <<'PY'
import json, math, sys
from pathlib import Path

path, patch_sha, mode, expected_updates, replacement = sys.argv[1:]
report = json.loads(Path(path).read_text(encoding="utf-8"))
expected_role = "smoke" if mode == "smoke" else "main"
expected_replacement = None if mode == "smoke" else replacement
expected_paper_eligible = mode == "main"
if (
    report.get("ok") is not True
    or report.get("run_role") != expected_role
    or report.get("training_seed") != 11
    or report.get("updates") != int(expected_updates)
    or report.get("paper_eligible") is not expected_paper_eligible
    or report.get("execution_patch_sha256") != patch_sha
    or report.get("replacement_of_job_id") != expected_replacement
    or report.get("prefetch_workers") != 7
    or report.get("prefetch_depth") != 14
    or not math.isfinite(float(report.get("walltime_s", float("nan"))))
):
    raise SystemExit(f"{mode} refiner acceptance failed")
checkpoint = Path(report["checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"{mode} refiner checkpoint is missing")
print(f"refiner_{mode}_acceptance=PASS")
PY

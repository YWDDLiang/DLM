#!/bin/bash
#SBATCH --job-name=wq-epoch-eval-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-2%2
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%A_%a.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%A_%a.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
REFINER="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26679_v1/model_ema_final.pt"
PANEL="$RUN/outputs/epoch_checkpoint_panel_v1"
EPOCH=$((SLURM_ARRAY_TASK_ID + 1))
EPOCH_PAD=$(printf '%02d' "$EPOCH")
ADAPTER="$TRAIN/epoch_${EPOCH_PAD}/adapter_final"
OUTPUT="$PANEL/epoch_${EPOCH_PAD}"

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ] \
  || [ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne 3 ]; then
  echo "epoch evaluation identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$TRAIN/epoch_${EPOCH_PAD}/training_report.json"
test -f "$REFINER"
test ! -e "$OUTPUT"
mkdir -p "$PANEL"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED=0

python - <<'PY'
import torch
if torch.cuda.device_count() != 1 or "A800" not in torch.cuda.get_device_name(0):
    raise SystemExit("epoch panel requires one A800 per task")
print(f"device={torch.cuda.get_device_name(0)}")
print("sun_protocol=R5-C A100 protocol on A800")
PY

mkdir "$OUTPUT"
usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python scripts/a800/evaluate_crysllmgen_lora_nll.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --adapter "$ADAPTER" \
  --data "$RUN/outputs/sft_wq_mixed_seed23_ctx512_v8/coarse_sft.jsonl" \
  --data-manifest "$RUN/outputs/sft_wq_mixed_seed23_ctx512_v8/coarse_sft.manifest.json" \
  --token-audit "$RUN/outputs/sft_wq_mixed_seed23_ctx512_v8/token_audit.json" \
  --representation wyckoff \
  --training-stage mixed_edit \
  --adapter-training-stage mixed_edit \
  --training-seed 11 \
  --data-seed 23 \
  --output "$OUTPUT/nll_report.json" \
  --batch-size 8 \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256"

python scripts/a800/sample_crysllmgen_wq.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --refiner-checkpoint "$REFINER" \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --llama-adapter "$ADAPTER" \
  --output-jsonl "$OUTPUT/generation.jsonl" \
  --attempt-ledger "$OUTPUT/attempt_ledger.jsonl" \
  --report "$OUTPUT/generation_report.json" \
  --experiment-id wq-epoch-selection-panel-v1 \
  --pairing-id wq-epoch-selection-paired-noise-v1 \
  --method C-WQ-HANDOFF \
  --training-seed 11 \
  --sampling-seed 101 \
  --attempts 256 \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256"

conda activate crysllm
export PYTHONPATH="$ROOT"
export CRYSLLMGEN_METRICS_NUM_CPUS=1
python scripts/a800/run_crysllmgen_metrics.py \
  --generation-jsonl "$OUTPUT/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
  --output-dir "$OUTPUT/crysllmgen_metrics" \
  --eval-model-name mp20

python scripts/a800/run_crysllmgen_a100_sun.py \
  --generation-jsonl "$OUTPUT/generation.jsonl" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/crysllm/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --output-dir "$OUTPUT/r5c_a100_sun"

python - "$OUTPUT" "$EPOCH" "$EXECUTION_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
epoch = int(sys.argv[2])
generation = [json.loads(line) for line in (root / "generation.jsonl").read_text().splitlines() if line]
nll = json.loads((root / "nll_report.json").read_text())
metrics = json.loads((root / "crysllmgen_metrics/report.json").read_text())
sun = json.loads((root / "r5c_a100_sun/attempt_summary.json").read_text())
if (
    len(generation) != 256
    or {row.get("method") for row in generation} != {"C-WQ-HANDOFF"}
    or any(row.get("retry_or_replacement_used") for row in generation)
    or nll.get("ok") is not True
    or nll.get("validation_data_seed") != 23
    or metrics.get("ok") is not True
    or metrics.get("attempts") != 256
    or sun.get("ok") is not True
    or (sun.get("counts") or {}).get("total_attempts") != 256
    or sun.get("execution_patch_sha256") != sys.argv[3]
):
    raise SystemExit(f"epoch {epoch} evaluation acceptance failed")
print(f"epoch_{epoch}_evaluation_acceptance=PASS")
PY

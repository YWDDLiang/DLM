#!/bin/bash
#SBATCH --job-name=wq-epoch-eval-sup27104-v1
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
EXPECTED_ADAPTER_TRAINING_PATCH=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491
EXPECTED_REFINER_TRAINING_PATCH=a868de4ac99d833628eca08304b6ca6066be58f138b5f03d1dd0e508a61beadc
EXPECTED_REFINER_SHA=8abb6ba966e0a7153fecf7141c97e659977692bc74ee4573317cbb26bddc4b70
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
REFINER_ROOT="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26955_v2"
REFINER="$REFINER_ROOT/model_ema_final.pt"
PANEL="$RUN/outputs/epoch_checkpoint_panel_sup27104_v1"
EPOCH=$((SLURM_ARRAY_TASK_ID + 1))
EPOCH_PAD=$(printf '%02d' "$EPOCH")
ADAPTER="$TRAIN/epoch_${EPOCH_PAD}/adapter_final"
TRAINING_REPORT="$TRAIN/epoch_${EPOCH_PAD}/training_report.json"
OUTPUT="$PANEL/epoch_${EPOCH_PAD}"

: "${EVALUATION_PATCH_SHA256:?submission must export EVALUATION_PATCH_SHA256}"
: "${ADAPTER_TRAINING_PATCH_SHA256:?submission must export ADAPTER_TRAINING_PATCH_SHA256}"
: "${REFINER_TRAINING_PATCH_SHA256:?submission must export REFINER_TRAINING_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$ADAPTER_TRAINING_PATCH_SHA256" != "$EXPECTED_ADAPTER_TRAINING_PATCH" ] \
  || [ "$REFINER_TRAINING_PATCH_SHA256" != "$EXPECTED_REFINER_TRAINING_PATCH" ] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ] \
  || [ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne 3 ]; then
  echo "epoch-evaluation supersession identity/resource contract changed" >&2
  exit 2
fi

case "$EPOCH" in
  1)
    EXPECTED_TRAINING_REPORT_SHA=f68a60de6909f5268c741816b4a754f704c3f22652be0676692174d19f616ad7
    EXPECTED_ADAPTER_SHA=6dd840a7380e117867feca077cf9ed58d17ce081e6466d25bac27b55fe969efc
    ;;
  2)
    EXPECTED_TRAINING_REPORT_SHA=23a9fe46b9b4ca75198c4c20b9b592a0f85e6e86b17b4b422cc287f678b49363
    EXPECTED_ADAPTER_SHA=398b4e175cd9a8dfa8f6e09a6bbb2c511522aca0a5c0a5298b4c9f2530ece18e
    ;;
  3)
    EXPECTED_TRAINING_REPORT_SHA=1d4bb3275dd5f1a20b9743161b59c633faad6f008ebbf7881ad07ddd2c6dfaf9
    EXPECTED_ADAPTER_SHA=9e12192fbd9fd42a1cdc236a50a3e70f8f97e4b44353cb37986c72dafd21b45c
    ;;
  *)
    echo "unregistered logical epoch" >&2
    exit 2
    ;;
esac

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${REFINER_TRAINING_PATCH_SHA256}.json"
test "$(sha256sum "$TRAINING_REPORT" | awk '{print $1}')" = "$EXPECTED_TRAINING_REPORT_SHA"
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = "$EXPECTED_ADAPTER_SHA"
test "$(sha256sum "$REFINER" | awk '{print $1}')" = "$EXPECTED_REFINER_SHA"
test "$(sha256sum "$REFINER_ROOT/training_report.json" | awk '{print $1}')" = 87e9b64a7154b8b451b118f18f80f4bcb575550cbef2b67fc0c9d507a9a70316
test ! -e "$OUTPUT"

python - "$TRAINING_REPORT" "$EPOCH" "$ADAPTER_TRAINING_PATCH_SHA256" \
  "$REFINER_ROOT/training_report.json" "$REFINER_TRAINING_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

training = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
refiner = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
if (
    training.get("schema") != "crysllmgen_lora_training_report_v1"
    or training.get("logical_epoch") != int(sys.argv[2])
    or training.get("global_step") != 848 * int(sys.argv[2])
    or training.get("execution_patch_sha256") != sys.argv[3]
    or training.get("slurm_job_id") != "26954"
    or refiner.get("schema") != "crysllmgen_wq_refiner_training_report_v1"
    or refiner.get("ok") is not True
    or refiner.get("paper_eligible") is not True
    or refiner.get("updates") != 100000
    or refiner.get("execution_patch_sha256") != sys.argv[5]
    or refiner.get("replacement_of_job_id") != "26955"
):
    raise SystemExit("reused training/refiner identity preflight failed")
print("reused_training_refiner_identity_preflight=PASS")
PY

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

mkdir -p "$PANEL"
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
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256"

python scripts/a800/sample_crysllmgen_wq.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --refiner-checkpoint "$REFINER" \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --llama-adapter "$ADAPTER" \
  --output-jsonl "$OUTPUT/generation.jsonl" \
  --attempt-ledger "$OUTPUT/attempt_ledger.jsonl" \
  --report "$OUTPUT/generation_report.json" \
  --experiment-id wq-epoch-selection-panel-sup27104-v1 \
  --pairing-id wq-epoch-selection-paired-noise-sup27104-v1 \
  --method C-WQ-HANDOFF \
  --training-seed 11 \
  --sampling-seed 101 \
  --attempts 256 \
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --refiner-training-execution-patch-sha256 "$REFINER_TRAINING_PATCH_SHA256" \
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256"

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
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --output-dir "$OUTPUT/r5c_a100_sun"

python - "$OUTPUT" "$EPOCH" "$ADAPTER_TRAINING_PATCH_SHA256" \
  "$REFINER_TRAINING_PATCH_SHA256" "$EVALUATION_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
epoch = int(sys.argv[2])
adapter_patch, refiner_patch, evaluation_patch = sys.argv[3:]
generation = [
    json.loads(line)
    for line in (root / "generation.jsonl").read_text().splitlines()
    if line
]
nll = json.loads((root / "nll_report.json").read_text())
sampling = json.loads((root / "generation_report.json").read_text())
metrics = json.loads((root / "crysllmgen_metrics/report.json").read_text())
sun = json.loads((root / "r5c_a100_sun/attempt_summary.json").read_text())
expected_identity = {
    (
        row.get("adapter_training_execution_patch_sha256"),
        row.get("refiner_training_execution_patch_sha256"),
        row.get("evaluation_execution_patch_sha256"),
    )
    for row in generation
}
if (
    len(generation) != 256
    or {row.get("method") for row in generation} != {"C-WQ-HANDOFF"}
    or any(row.get("retry_or_replacement_used") for row in generation)
    or expected_identity != {(adapter_patch, refiner_patch, evaluation_patch)}
    or nll.get("ok") is not True
    or nll.get("validation_data_seed") != 23
    or nll.get("adapter_training_execution_patch_sha256") != adapter_patch
    or nll.get("evaluation_execution_patch_sha256") != evaluation_patch
    or sampling.get("ok") is not True
    or (sampling.get("model_identity") or {}).get(
        "refiner_training_execution_patch_sha256"
    ) != refiner_patch
    or (sampling.get("model_identity") or {}).get(
        "adapter_training_execution_patch_sha256"
    ) != adapter_patch
    or (sampling.get("model_identity") or {}).get(
        "evaluation_execution_patch_sha256"
    ) != evaluation_patch
    or metrics.get("ok") is not True
    or metrics.get("attempts") != 256
    or sun.get("ok") is not True
    or (sun.get("counts") or {}).get("total_attempts") != 256
    or sun.get("execution_patch_sha256") != evaluation_patch
):
    raise SystemExit(f"epoch {epoch} evaluation supersession acceptance failed")
print(f"epoch_{epoch}_evaluation_supersession_acceptance=PASS")
PY

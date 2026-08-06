#!/bin/bash
#SBATCH --job-name=wq-lora-3ep-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ID=20260720_0401-crysllmgen-wq-final-v3
RUN="runs/$RUN_ID"
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
PROTOCOL_SHA=22efb63d3b7e37353206022d3ce9e43f0c19ed02dbd5fa75a71a053339707af0
AMENDMENT=configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json
AMENDMENT_SHA=af6e72ce842d5c604ebd80f4d90c31edea55f406a1c2da0ea41e89adfcf96afd
FLASH_MATRIX=configs/experiments/wyckoff_codiffusion/ddp_flash_followup_v1.json
FLASH_MATRIX_SHA=3d2f7276d04f9c36ea3c962d52d2e553ee0dd2f79ed445625aae6785337dbfc3
FLASH_PROFILE_ID=cache4_mb8_acc4_gc_off_flash2
FLASH_OUTPUT="$RUN/outputs/ddp_flash_followup_v1/$FLASH_PROFILE_ID"
SELECTION="$RUN/notes/formal_execution_selection_v1.json"
CACHE="$RUN/outputs/token_cache_wq_mixed_seed11_ctx512_profile_matrix_v1"
TRAIN_OUTPUT="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
FLASH_ENV=/public/home/jiaosz/ywliang/ai4s/.venvs/flash_attn_2_7_4_post1_torch2_4_cu12_cp310_abi0_glibc214

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
: "${FLASH_PROFILE_JOB_ID:?submission must export FLASH_PROFILE_JOB_ID}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$FLASH_PROFILE_JOB_ID" =~ ^[0-9]+$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]; then
  echo "formal training identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test "$(sha256sum "$AMENDMENT" | awk '{print $1}')" = "$AMENDMENT_SHA"
test "$(sha256sum "$FLASH_MATRIX" | awk '{print $1}')" = "$FLASH_MATRIX_SHA"
test "$(sha256sum "$CACHE/cache_manifest.json" | awk '{print $1}')" = \
  afd954868f1f9c431068e92b444061079d741d6b23e8cdddccb1e9bedd9ddb23
test ! -e "$SELECTION"
test ! -e "$TRAIN_OUTPUT"
mkdir -p "$RUN/notes"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONHASHSEED=0

python scripts/a800/select_crysllmgen_execution.py \
  --run-id "$RUN_ID" \
  --flash-job-id "$FLASH_PROFILE_JOB_ID" \
  --flash-profile-matrix "$FLASH_MATRIX" \
  --flash-report "$FLASH_OUTPUT/performance_profile.json" \
  --flash-stderr "$RUN/logs/wq-fa2-prof-v1-${FLASH_PROFILE_JOB_ID}.err" \
  --flash-gpu-csv "$RUN/logs/wq-fa2-prof-v1-${FLASH_PROFILE_JOB_ID}.gpu.csv" \
  --source-bundle-sha256 "$BASE_SOURCE" \
  --protocol-v4-sha256 "$PROTOCOL_SHA" \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --output "$SELECTION"

ATTENTION=$(python - "$SELECTION" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_attention_implementation"])
PY
)
if [ "$ATTENTION" = flash_attention_2 ]; then
  conda deactivate
  source "$FLASH_ENV/bin/activate"
  python - <<'PY'
import flash_attn
if flash_attn.__version__ != "2.7.4.post1":
    raise SystemExit("formal FlashAttention environment changed")
PY
elif [ "$ATTENTION" != sdpa ]; then
  echo "unregistered attention selection: $ATTENTION" >&2
  exit 2
fi

python - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected two visible GPUs, got {torch.cuda.device_count()}")
if any("A800" not in torch.cuda.get_device_name(index) for index in range(2)):
    raise SystemExit("formal training requires two A800 devices")
print(f"devices={[torch.cuda.get_device_name(index) for index in range(2)]}")
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

torchrun --standalone --nproc_per_node=2 scripts/a800/train_crysllmgen_lora.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --training-amendment "$AMENDMENT" \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --data "$RUN/outputs/sft_wq_mixed_seed11_ctx512_v3/coarse_sft.jsonl" \
  --data-manifest "$RUN/outputs/sft_wq_mixed_seed11_ctx512_v3/coarse_sft.manifest.json" \
  --token-audit "$RUN/outputs/sft_wq_mixed_seed11_ctx512_v3/token_audit.json" \
  --representation wyckoff \
  --training-stage mixed_edit \
  --initial-adapter "$RUN/outputs/train_wq_lora_seed11_coarse_v8/adapter_final" \
  --training-seed 11 \
  --output-dir "$TRAIN_OUTPUT" \
  --source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --formal-execution-selection "$SELECTION" \
  --run-id "$RUN_ID" \
  --pretokenized-cache "$CACHE" \
  --run-role main

python - "$TRAIN_OUTPUT/training_report.json" "$ATTENTION" \
  "$EXECUTION_PATCH_SHA256" "$AMENDMENT_SHA" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
optimizer = report.get("optimizer") or {}
execution = report.get("execution") or {}
checkpoints = report.get("epoch_checkpoints") or []
if (
    report.get("run_role") != "main"
    or report.get("execution_patch_sha256") != sys.argv[3]
    or (report.get("training_amendment") or {}).get("sha256") != sys.argv[4]
    or optimizer.get("completed_global_step") != 2544
    or optimizer.get("per_device_microbatch") != 8
    or optimizer.get("gradient_accumulation") != 4
    or execution.get("attention_implementation") != sys.argv[2]
    or execution.get("gradient_checkpointing") is not False
    or [item.get("global_step") for item in checkpoints] != [848, 1696, 2544]
):
    raise SystemExit("formal three-epoch training acceptance failed")
print("formal_three_epoch_training_acceptance=PASS")
PY

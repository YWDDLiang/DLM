#!/bin/bash
#SBATCH --job-name=wq-fa2-prof-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
PROTOCOL_SHA=22efb63d3b7e37353206022d3ce9e43f0c19ed02dbd5fa75a71a053339707af0
AMENDMENT=configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json
AMENDMENT_SHA=af6e72ce842d5c604ebd80f4d90c31edea55f406a1c2da0ea41e89adfcf96afd
MATRIX=configs/experiments/wyckoff_codiffusion/ddp_flash_followup_v1.json
MATRIX_SHA=3d2f7276d04f9c36ea3c962d52d2e553ee0dd2f79ed445625aae6785337dbfc3
PROFILE_ID=cache4_mb8_acc4_gc_off_flash2
CACHE="$RUN/outputs/token_cache_wq_mixed_seed11_ctx512_profile_matrix_v1"
OUTPUT="$RUN/outputs/ddp_flash_followup_v1/$PROFILE_ID"
FLASH_ENV=/public/home/jiaosz/ywliang/ai4s/.venvs/flash_attn_2_7_4_post1_torch2_4_cu12_cp310_abi0_glibc214

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]; then
  echo "FlashAttention profile identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test "$(sha256sum "$AMENDMENT" | awk '{print $1}')" = "$AMENDMENT_SHA"
test "$(sha256sum "$MATRIX" | awk '{print $1}')" = "$MATRIX_SHA"
test "$(sha256sum "$CACHE/cache_manifest.json" | awk '{print $1}')" = \
  afd954868f1f9c431068e92b444061079d741d6b23e8cdddccb1e9bedd9ddb23
test ! -e "$OUTPUT"
mkdir -p "$RUN/outputs/ddp_flash_followup_v1"

source "$FLASH_ENV/bin/activate"
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

python - <<'PY'
import flash_attn
import torch

if flash_attn.__version__ != "2.7.4.post1":
    raise SystemExit(f"unexpected flash-attn version: {flash_attn.__version__}")
if torch.__version__ != "2.4.0+cu121":
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected two visible GPUs, got {torch.cuda.device_count()}")
if any("A800" not in torch.cuda.get_device_name(index) for index in range(2)):
    raise SystemExit("both allocated devices must be A800")
print(f"flash_attn={flash_attn.__version__} torch={torch.__version__}")
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
  --output-dir "$OUTPUT" \
  --source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --performance-profile-matrix "$MATRIX" \
  --performance-profile-id "$PROFILE_ID" \
  --pretokenized-cache "$CACHE" \
  --max-steps 50 \
  --run-role performance_profile

python - "$OUTPUT/performance_profile.json" "$PROTOCOL_SHA" \
  "$EXECUTION_PATCH_SHA256" "$MATRIX_SHA" <<'PY'
import json, math, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
optimizer = report.get("optimizer") or {}
execution = report.get("execution") or {}
runtime = report.get("runtime") or {}
if (
    report.get("pass") is not True
    or report.get("scientific_attempt") is not False
    or report.get("protocol_sha256") != sys.argv[2]
    or report.get("execution_patch_sha256") != sys.argv[3]
    or (report.get("profile_matrix") or {}).get("sha256") != sys.argv[4]
    or optimizer.get("completed_global_step") != 50
    or optimizer.get("global_effective_batch") != 64
    or execution.get("attention_implementation") != "flash_attention_2"
    or execution.get("gradient_checkpointing") is not False
    or not math.isfinite(float(runtime.get("train_seconds_per_optimizer_update", float("nan"))))
):
    raise SystemExit("FlashAttention performance-profile acceptance failed")
print("flash_attention_profile_acceptance=PASS")
PY

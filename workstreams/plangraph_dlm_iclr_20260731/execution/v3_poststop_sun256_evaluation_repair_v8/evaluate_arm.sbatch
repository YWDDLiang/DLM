#!/bin/bash
#SBATCH --job-name=v3-sun256-v8-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_evaluation_repair_v8"
RUN="$ROOT/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8"
V7_RUN="$ROOT/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v7"
V7_SOURCE_SHA=5d05e23da6ba4e0e49f4646a5db8181c27f3dd47d67148cac9baaa842fa6a42f
: "${V8_SUN_SOURCE_SHA256:?v8 source manifest SHA is required}"

arms=(M00 M10 M01 M11)
ARM="${arms[${SLURM_ARRAY_TASK_ID:?array task ID is required}]}"
ARM_ROOT="$RUN/arms/$ARM"
GENERATION_JSONL="$V7_RUN/arms/$ARM/generation/generation.jsonl"
FROZEN_RUNTIME="$SOURCE/runtime"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$V8_SUN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "evaluation-only identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$FROZEN_RUNTIME:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

test ! -e "$ARM_ROOT"
echo "STAGE arm=$ARM frozen_input_and_runtime_preflight"
python "$SOURCE/preflight_arm.py" \
  --arm "$ARM" \
  --source-dir "$SOURCE" \
  --input-manifest "$SOURCE/input_manifest.json" \
  --output-dir "$ARM_ROOT" \
  --source-manifest-sha256 "$V8_SUN_SOURCE_SHA256"

python - <<'PY'
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

echo "STAGE arm=$ARM frozen_cache_sun"
python "$FROZEN_RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
  --generation-jsonl "$GENERATION_JSONL" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 "$V7_SOURCE_SHA" \
  --execution-patch-sha256 "$V8_SUN_SOURCE_SHA256" \
  --output-dir "$ARM_ROOT/r5c_a100_sun"

echo "STAGE arm=$ARM validate_all_attempt_mapping"
python "$SOURCE/validate_arm.py" \
  --arm "$ARM" \
  --source-dir "$SOURCE" \
  --input-manifest "$SOURCE/input_manifest.json" \
  --output-dir "$ARM_ROOT" \
  --source-manifest-sha256 "$V8_SUN_SOURCE_SHA256"

test -f "$ARM_ROOT/_SUCCESS"
echo "STAGE arm=$ARM complete"

#!/bin/bash
#SBATCH --job-name=h1a2c-sun256-eval-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-1%2
#SBATCH --output=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%A_%a.out
#SBATCH --error=runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/logs/%x-%A_%a.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1"
RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1"
: "${H1A2C_SUN256_SOURCE_SHA256:?source manifest SHA256 is required}"
arms=(P0 P1)
ARM="${arms[${SLURM_ARRAY_TASK_ID:?array task ID is required}]}"
GEN="$RUN/arms/$ARM/generation"
EVAL="$RUN/arms/$ARM/evaluation"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1A2C_SUN256_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "evaluation identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CRYSLLMGEN_METRICS_NUM_CPUS=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("S.U.N. evaluation requires exactly one CUDA device")
device_name = torch.cuda.get_device_name(0)
if "A800" not in device_name:
    raise RuntimeError(f"S.U.N. evaluation requires A800, observed {device_name}")
PY

test -f "$GEN/_SUCCESS"
test ! -e "$EVAL"
mkdir "$EVAL"

python scripts/a800/run_crysllmgen_metrics.py \
  --generation-jsonl "$GEN/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
  --output-dir "$EVAL/crysllmgen_metrics" \
  --eval-model-name mp20

python scripts/a800/run_crysllmgen_a100_sun.py \
  --generation-jsonl "$GEN/generation.jsonl" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 "$H1A2C_SUN256_SOURCE_SHA256" \
  --execution-patch-sha256 "$H1A2C_SUN256_SOURCE_SHA256" \
  --output-dir "$EVAL/r5c_a100_sun"

python "$SOURCE/validate_evaluation.py" \
  --arm "$ARM" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --data-dir "$RUN/data" \
  --generation-dir "$GEN" \
  --output-dir "$EVAL" \
  --execution-manifest-sha256 "$H1A2C_SUN256_SOURCE_SHA256"

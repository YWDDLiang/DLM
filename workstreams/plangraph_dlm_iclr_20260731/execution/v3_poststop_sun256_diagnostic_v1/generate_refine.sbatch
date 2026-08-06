#!/bin/bash
#SBATCH --job-name=v3-sun256-genref
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/v3_poststop_sun256_diagnostic_v1"
RUN="$ROOT/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1"
BASE_MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
FROZEN_RUNTIME="$SOURCE/runtime"
VOCAB_SHA=dd3968132cdd9f75f92363a842d26a8d083a4a05f0411ee856ba5e473090815d
: "${V3_SUN_SOURCE_SHA256:?execution source manifest SHA is required}"

arms=(M00 M10 M01 M11)
ARM="${arms[${SLURM_ARRAY_TASK_ID:?array task ID is required}]}"
case "$ARM" in
  M00)
    PLANNER=P0
    BODY_CHECKPOINT="$ROOT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
    BODY_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
    ;;
  M10)
    PLANNER=Pstar
    BODY_CHECKPOINT="$ROOT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
    BODY_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
    ;;
  M01)
    PLANNER=P0
    BODY_CHECKPOINT="$ROOT/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1/arms/B2/output/final"
    BODY_SHA=e3451c94822ca83d1fa1f91680e24880ce9ced022379865b76705554f31f8cb8
    ;;
  M11)
    PLANNER=Pstar
    BODY_CHECKPOINT="$ROOT/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1/arms/B2/output/final"
    BODY_SHA=e3451c94822ca83d1fa1f91680e24880ce9ced022379865b76705554f31f8cb8
    ;;
  *)
    exit 2
    ;;
esac

ARM_ROOT="$RUN/arms/$ARM"
BODY_DIR="$ARM_ROOT/body"
REFINE_DIR="$ARM_ROOT/refinement"
GEN_DIR="$ARM_ROOT/generation"
PLANNER_ATTEMPTS="$RUN/data/$PLANNER/planner_attempts.jsonl"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$V3_SUN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "generation/refinement identity/resource contract changed" >&2
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

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

test -f "$RUN/data/_SUCCESS"
test -f "$PLANNER_ATTEMPTS"
test -d "$BODY_CHECKPOINT"
test -f "$REFINER"
test ! -e "$ARM_ROOT"
mkdir -p "$ARM_ROOT"

python "$FROZEN_RUNTIME/scripts/sample_llada_h1a2_factorial_body.py" \
  --model-path "$BASE_MODEL" \
  --checkpoint-path "$BODY_CHECKPOINT" \
  --checkpoint-sha256 "$BODY_SHA" \
  --planner-attempts "$PLANNER_ATTEMPTS" \
  --factorial-arm "$ARM" \
  --output-dir "$BODY_DIR" \
  --crysllmgen-dir "$FROZEN_RUNTIME/reference/crysllmgen" \
  --num-samples 256 \
  --base-seed 17029 \
  --expected-tokenizer-vocab-sha256 "$VOCAB_SHA" \
  --temperature 0.7 \
  --cfg-scale 0.0 \
  --remasking low_confidence \
  --min-lattice-rad 0.0001

python "$FROZEN_RUNTIME/scripts/refine_h1a2_factorial_with_crysllmgen.py" \
  --body-attempts "$BODY_DIR/body_attempts.jsonl" \
  --proposal-graphs "$BODY_DIR/proposal_graphs.pt" \
  --factorial-arm "$ARM" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$REFINE_DIR" \
  --crysllmgen-dir "$FROZEN_RUNTIME/reference/crysllmgen" \
  --num-samples 256

python "$SOURCE/finalize_generation.py" \
  --arm "$ARM" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --refinement-dir "$REFINE_DIR" \
  --output-dir "$GEN_DIR" \
  --execution-manifest-sha256 "$V3_SUN_SOURCE_SHA256"

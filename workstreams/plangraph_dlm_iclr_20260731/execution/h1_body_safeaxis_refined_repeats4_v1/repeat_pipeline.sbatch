#!/bin/bash
#SBATCH --job-name=h1-r03e-repeat
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUN="$ROOT/runs/20260802_h1_body_safeaxis_refined_repeats4_v1"
CONFIG="$SOURCE/config.json"
R03D_RUN="$ROOT/runs/20260802_h1_body_safeaxis256_v1"
LEDGER="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
RUNTIME="$SOURCE/runtime"
: "${R03E_SOURCE_SHA256:?R03E source manifest SHA is required}"

REPEAT="${SLURM_ARRAY_TASK_ID:?array task ID is required}"
case "$REPEAT" in
  0|3) ORDER=(control candidate) ;;
  1|2) ORDER=(candidate control) ;;
  *) echo "invalid repeat" >&2; exit 2 ;;
esac
REPEAT_ROOT="$RUN/repeats/$REPEAT"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$R03E_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "R03E identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$RUNTIME:$SOURCE"
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
export CRYSLLMGEN_METRICS_NUM_CPUS=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

echo "STAGE repeat=$REPEAT frozen_input_and_runtime_preflight"
python "$SOURCE/preflight_repeat.py" \
  --repeat "$REPEAT" \
  --config "$CONFIG" \
  --source-dir "$SOURCE" \
  --output-dir "$REPEAT_ROOT" \
  --source-manifest-sha256 "$R03E_SOURCE_SHA256"

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

for ARM in "${ORDER[@]}"; do
  case "$ARM" in
    control)
      BODY_ATTEMPTS="$R03D_RUN/screen/control_body_attempts.jsonl"
      PROPOSAL_GRAPHS="$R03D_RUN/screen/control_proposal_graphs.pt"
      ;;
    candidate)
      BODY_ATTEMPTS="$R03D_RUN/screen/candidate_body_attempts.jsonl"
      PROPOSAL_GRAPHS="$R03D_RUN/screen/candidate_proposal_graphs.pt"
      ;;
    *) exit 2 ;;
  esac
  ARM_ROOT="$REPEAT_ROOT/arms/$ARM"
  REFINE_DIR="$ARM_ROOT/refinement"
  GENERATION_DIR="$ARM_ROOT/generation"
  EVALUATION_DIR="$ARM_ROOT/evaluation"

  echo "STAGE repeat=$REPEAT arm=$ARM frozen_refine800"
  python "$SOURCE/refine_safeaxis_arm.py" \
    --arm "$ARM" \
    --repeat "$REPEAT" \
    --config "$CONFIG" \
    --body-attempts "$BODY_ATTEMPTS" \
    --proposal-graphs "$PROPOSAL_GRAPHS" \
    --attempt-ledger "$LEDGER" \
    --checkpoint "$REFINER" \
    --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
    --output-dir "$REFINE_DIR" \
    --crysllmgen-dir "$RUNTIME/reference/crysllmgen"

  echo "STAGE repeat=$REPEAT arm=$ARM finalize_all_attempt"
  python "$SOURCE/finalize_generation.py" \
    --arm "$ARM" \
    --repeat "$REPEAT" \
    --config "$CONFIG" \
    --source-dir "$SOURCE" \
    --source-manifest-sha256 "$R03E_SOURCE_SHA256" \
    --refinement-dir "$REFINE_DIR" \
    --output-dir "$GENERATION_DIR"

  test -f "$GENERATION_DIR/_SUCCESS"
  test ! -e "$EVALUATION_DIR"
  mkdir "$EVALUATION_DIR"

  echo "STAGE repeat=$REPEAT arm=$ARM direct_metrics"
  python "$RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "$GENERATION_DIR/generation.jsonl" \
    --gt-csv reference/crysllmgen/data/mp_20/test.csv \
    --snapshot-root "$RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "$EVALUATION_DIR/crysllmgen_metrics" \
    --eval-model-name mp20

  echo "STAGE repeat=$REPEAT arm=$ARM frozen_cache_sun"
  python "$RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
    --generation-jsonl "$GENERATION_DIR/generation.jsonl" \
    --expected-attempts 256 \
    --eval-sun-py reference/a100_eval_sun/eval_sun.py \
    --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
    --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
    --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
    --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
    --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
    --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
    --execution-patch-sha256 "$R03E_SOURCE_SHA256" \
    --output-dir "$EVALUATION_DIR/r5c_a100_sun"

  echo "STAGE repeat=$REPEAT arm=$ARM validate"
  python "$SOURCE/validate_evaluation.py" \
    --arm "$ARM" \
    --repeat "$REPEAT" \
    --config "$CONFIG" \
    --source-dir "$SOURCE" \
    --source-manifest-sha256 "$R03E_SOURCE_SHA256" \
    --generation-dir "$GENERATION_DIR" \
    --output-dir "$EVALUATION_DIR"
  test -f "$EVALUATION_DIR/_SUCCESS"
done

touch "$REPEAT_ROOT/_SUCCESS"
echo "STAGE repeat=$REPEAT complete"

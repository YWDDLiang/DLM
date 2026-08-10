#!/bin/bash
#SBATCH --job-name=h1-r03p-b3-r4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_v1/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_v1/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_v1"
SOURCE="$RUN_ROOT/source"
V4_SOURCE="$PROJECT/workstreams/final_method_development_20260808/execution/h1_four_cell_direct_sun_v4"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
R03E_RUNTIME="$R03E/runtime"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
: "${H1_R03B3_SOURCE_SHA256:?source manifest SHA256 is required}"

case "${SLURM_ARRAY_TASK_ID:?array task is required}" in
  0) CELL=M00; REPEAT=0 ;;
  1) CELL=M10; REPEAT=1 ;;
  2) CELL=M01; REPEAT=2 ;;
  3) CELL=M11; REPEAT=3 ;;
  *) echo "invalid R03/B3 repeat array task" >&2; exit 2 ;;
esac
CELL_ROOT="$RUN_ROOT/cells/$CELL"
STATUS="$RUN_ROOT/status/repeat_${REPEAT}_exit_code.txt"
test ! -e "$STATUS"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$STATUS"
  if [[ "$rc" -ne 0 ]] && [[ -d "$CELL_ROOT" ]]; then
    touch "$CELL_ROOT/_FAILURE"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1_R03B3_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_R03B3_SOURCE_SHA256" ]] \
  || [[ "$(sha256sum "$V4_SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != cb9391a919bea71a8a96c4fb576901101aed6a40069b89f27c16fc38ea9c518d ]]; then
  echo "R03/B3 resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
(cd "$V4_SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/preparation_SUCCESS"
test ! -e "$CELL_ROOT"
mkdir "$CELL_ROOT"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CRYSLLMGEN_METRICS_NUM_CPUS=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

python - <<'PY'
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

echo "STAGE repeat=$REPEAT cell=$CELL safe_axis_body256"
export PYTHONPATH="$V4_SOURCE:$R03D/runtime:$R03D"
python "$V4_SOURCE/run_body_safeaxis.py" \
  --cell "$CELL" \
  --config "$SOURCE/CONFIG.json" \
  --output-dir "$CELL_ROOT/body"

echo "STAGE repeat=$REPEAT cell=$CELL frozen_refine800"
export PYTHONPATH="$V4_SOURCE:$R03E_RUNTIME:$R03E"
python "$V4_SOURCE/refine_cell.py" \
  --cell "$CELL" \
  --config "$SOURCE/CONFIG.json" \
  --body-attempts "$CELL_ROOT/body/body_attempts.jsonl" \
  --proposal-graphs "$CELL_ROOT/body/proposal_graphs.pt" \
  --attempt-ledger "$PROJECT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$CELL_ROOT/refinement" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen"

echo "STAGE repeat=$REPEAT cell=$CELL finalize_all_attempt"
python "$V4_SOURCE/finalize_generation.py" \
  --cell "$CELL" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03B3_SOURCE_SHA256" \
  --refinement-dir "$CELL_ROOT/refinement" \
  --output-dir "$CELL_ROOT/generation"

mkdir "$CELL_ROOT/evaluation"
echo "STAGE repeat=$REPEAT cell=$CELL direct_metrics"
python "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
  --generation-jsonl "$CELL_ROOT/generation/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root "$R03E_RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "$CELL_ROOT/evaluation/crysllmgen_metrics" \
  --eval-model-name mp20

MP_CACHE="$RUN_ROOT/mp_cache/r03f_common_mp_thermo_snapshot.jsonl"
MP_CACHE_SHA_FILE="$RUN_ROOT/mp_cache/r03f_common_mp_thermo_snapshot.sha256"
[[ -f "$MP_CACHE" && -f "$MP_CACHE_SHA_FILE" ]]
MP_CACHE_SHA="$(cut -d' ' -f1 "$MP_CACHE_SHA_FILE")"
test "$MP_CACHE_SHA" = 56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917
test "$MP_CACHE_SHA" = "$(sha256sum "$MP_CACHE" | cut -d' ' -f1)"

echo "STAGE repeat=$REPEAT cell=$CELL r03f_cache_sun"
python "$V4_SOURCE/run_crysllmgen_a100_sun_completed_cache.py" \
  --runner "$R03E_RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
  --runner-sha256 e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de \
  --a100-sun-module-sha256 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968 \
  --expected-mp-hull-cache-sha256 "$MP_CACHE_SHA" \
  -- \
  --generation-jsonl "$CELL_ROOT/generation/generation.jsonl" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache "$MP_CACHE" \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
  --execution-patch-sha256 "$H1_R03B3_SOURCE_SHA256" \
  --output-dir "$CELL_ROOT/evaluation/r5c_a100_sun"

echo "STAGE repeat=$REPEAT cell=$CELL validate"
python "$V4_SOURCE/validate_cell.py" \
  --cell "$CELL" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03B3_SOURCE_SHA256" \
  --generation-dir "$CELL_ROOT/generation" \
  --output-dir "$CELL_ROOT/evaluation"

touch "$CELL_ROOT/_SUCCESS"
echo "STAGE repeat=$REPEAT cell=$CELL complete"

#!/bin/bash
#SBATCH --job-name=h1-native1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/native-%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/native-%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
BODY_SOURCE="$RUN_ROOT/body_source"
NATIVE_SOURCE="$RUN_ROOT/native1000_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
R03E_RUNTIME="$R03E/runtime"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_NATIVE1000_SOURCE_SHA256:?native source manifest SHA is required}"
: "${H1_ARM:?R03 or B3 arm is required}"

case "$H1_ARM" in
  R03|B3) ARM="$H1_ARM" ;;
  *) echo "invalid native body arm" >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID:?array task is required}" in
  0) REPEAT=0 ;;
  1) REPEAT=1 ;;
  2) REPEAT=2 ;;
  *) echo "invalid native repeat" >&2; exit 2 ;;
esac

MAIN_ROOT="$RUN_ROOT/arms/$ARM/repeats/$REPEAT"
REPEAT_ROOT="$RUN_ROOT/crysllmgen_native1000/arms/$ARM/repeats/$REPEAT"
CANDIDATE_ROOT="$RUN_ROOT/repeats/$REPEAT/crysllmgen_native_candidates"
STATUS="$RUN_ROOT/status/native1000_${ARM}_repeat_${REPEAT}_exit_code.txt"
test ! -e "$STATUS"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$STATUS"
  if [[ "$rc" -ne 0 ]] && [[ -d "$REPEAT_ROOT" ]]; then
    touch "$REPEAT_ROOT/_FAILURE"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1_NATIVE1000_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "$(sha256sum "$NATIVE_SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_NATIVE1000_SOURCE_SHA256" ]]; then
  echo "native resource or source identity changed" >&2
  exit 2
fi
(cd "$NATIVE_SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/native1000_preparation_SUCCESS"
test -f "$RUN_ROOT/native_mp_cache/completion_SUCCESS"
test -f "$CANDIDATE_ROOT/_SUCCESS"
test -f "$MAIN_ROOT/_SUCCESS"
test ! -e "$REPEAT_ROOT"
mkdir -p "$RUN_ROOT/crysllmgen_native1000/arms/$ARM/repeats"
mkdir "$REPEAT_ROOT"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/native1000_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
export PYTHONPATH="$NATIVE_SOURCE:$BODY_SOURCE:$R03D/runtime:$R03D:$R03E_RUNTIME:$R03E"

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected exactly one A800, observed {names}")
version = metadata.version("smact")
if not version.startswith("3.1"):
    raise SystemExit(f"SMACT 3.1 required, observed {version}")
print({"gpu": names, "torch": torch.__version__, "smact": version})
PY

echo "STAGE native1000 arm=$ARM repeat=$REPEAT reserve_safe_axis_body"
"$PYTHON" "$NATIVE_SOURCE/run_reserve_safeaxis.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --candidate-pool "$CANDIDATE_ROOT/candidate_pool.jsonl" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen" \
  --body-config "$BODY_SOURCE/CONFIG.json" \
  --output-dir "$REPEAT_ROOT/reserve_body"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT select_first1000_body_successes"
"$PYTHON" "$NATIVE_SOURCE/assemble_native_body1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --candidate-pool "$CANDIDATE_ROOT/candidate_pool.jsonl" \
  --main-body-dir "$MAIN_ROOT/body" \
  --reserve-body-dir "$REPEAT_ROOT/reserve_body" \
  --output-dir "$REPEAT_ROOT/selected_body1000"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT refine_selected_reserve"
"$PYTHON" "$NATIVE_SOURCE/refine_selected_reserve.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --body-config "$BODY_SOURCE/CONFIG.json" \
  --selected-body-attempts "$REPEAT_ROOT/selected_body1000/body_attempts.jsonl" \
  --selected-proposal-graphs "$REPEAT_ROOT/selected_body1000/proposal_graphs.pt" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$REPEAT_ROOT/reserve_refinement" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT finalize_full1000_post_model494"
"$PYTHON" "$NATIVE_SOURCE/finalize_native_post1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --selected-body-dir "$REPEAT_ROOT/selected_body1000" \
  --main-post-generation-dir "$MAIN_ROOT/stages/post_model494/generation" \
  --reserve-refinement-dir "$REPEAT_ROOT/reserve_refinement" \
  --output-dir "$REPEAT_ROOT/post_model494/generation"

GENERATION="$REPEAT_ROOT/post_model494/generation"
EVALUATION="$REPEAT_ROOT/post_model494/evaluation"
mkdir -p "$EVALUATION"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT direct_crysllmgen"
"$PYTHON" "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
  --generation-jsonl "$GENERATION/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root "$R03E_RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "$EVALUATION/crysllmgen_metrics" \
  --eval-model-name mp20

CACHE="$RUN_ROOT/native_mp_cache/completed_mp_hull_cache.jsonl"
CACHE_SHA="$("$PYTHON" - "$RUN_ROOT/native_mp_cache/completion_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["completed_mp_hull_cache"]["sha256"])
PY
)"
test "$CACHE_SHA" = "$(sha256sum "$CACHE" | cut -d' ' -f1)"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT complete_cache_sun"
"$PYTHON" "$BODY_SOURCE/run_crysllmgen_a100_sun_completed_cache.py" \
  --runner "$R03E_RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
  --runner-sha256 e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de \
  --a100-sun-module-sha256 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968 \
  --expected-mp-hull-cache-sha256 "$CACHE_SHA" \
  -- \
  --generation-jsonl "$GENERATION/generation.jsonl" \
  --expected-attempts 1000 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache "$CACHE" \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
  --execution-patch-sha256 "$H1_NATIVE1000_SOURCE_SHA256" \
  --output-dir "$EVALUATION/r5c_a100_sun"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT validate_evaluation"
"$PYTHON" "$BODY_SOURCE/validate_stage1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --stage post_model494 \
  --config "$BODY_SOURCE/CONFIG.json" \
  --source-dir "$NATIVE_SOURCE" \
  --source-manifest-sha256 "$H1_NATIVE1000_SOURCE_SHA256" \
  --generation-dir "$GENERATION" \
  --evaluation-dir "$EVALUATION" \
  --completion-manifest "$RUN_ROOT/native_mp_cache/completion_manifest.json"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT validate_complete1000"
"$PYTHON" "$NATIVE_SOURCE/validate_native1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --repeat-root "$REPEAT_ROOT"

echo "STAGE native1000 arm=$ARM repeat=$REPEAT complete"

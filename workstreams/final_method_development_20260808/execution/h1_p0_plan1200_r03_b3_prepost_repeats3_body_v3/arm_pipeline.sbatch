#!/bin/bash
#SBATCH --job-name=h1-plan1200-body
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/body-%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/body-%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/body_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
R03E_RUNTIME="$R03E/runtime"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_PLAN1200_BODY_SOURCE_SHA256:?body source manifest SHA is required}"
: "${H1_ARM:?R03 or B3 arm is required}"

case "$H1_ARM" in
  R03|B3) ARM="$H1_ARM" ;;
  *) echo "invalid body arm" >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID:?array task is required}" in
  0) REPEAT=0 ;;
  1) REPEAT=1 ;;
  2) REPEAT=2 ;;
  *) echo "invalid body repeat" >&2; exit 2 ;;
esac

REPEAT_ROOT="$RUN_ROOT/arms/$ARM/repeats/$REPEAT"
COHORT="$RUN_ROOT/repeats/$REPEAT/cohort/cohort1000.jsonl"
STATUS="$RUN_ROOT/status/body_${ARM}_repeat_${REPEAT}_exit_code.txt"
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
  || [[ ! "$H1_PLAN1200_BODY_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_PLAN1200_BODY_SOURCE_SHA256" ]]; then
  echo "body resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/body_preparation_SUCCESS"
test -f "$RUN_ROOT/status/body_preflight_report.json"
test -f "$RUN_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/mp_cache/completion_SUCCESS"
test -f "$COHORT"
mkdir -p "$RUN_ROOT/arms/$ARM/repeats"
test ! -e "$REPEAT_ROOT"
mkdir "$REPEAT_ROOT"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/body_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

echo "STAGE arm=$ARM repeat=$REPEAT safe_axis_body1000"
export PYTHONPATH="$SOURCE:$R03D/runtime:$R03D"
"$PYTHON" "$SOURCE/run_body_safeaxis1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --cohort "$COHORT" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen" \
  --config "$SOURCE/CONFIG.json" \
  --output-dir "$REPEAT_ROOT/body"

evaluate_stage() {
  local stage="$1"
  local stage_root="$REPEAT_ROOT/stages/$stage"
  local generation="$stage_root/generation"
  local evaluation="$stage_root/evaluation"
  mkdir -p "$evaluation"

  echo "STAGE arm=$ARM repeat=$REPEAT stage=$stage direct_crysllmgen"
  export PYTHONPATH="$SOURCE:$R03E_RUNTIME:$R03E"
  "$PYTHON" "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "$generation/generation.jsonl" \
    --gt-csv reference/crysllmgen/data/mp_20/test.csv \
    --snapshot-root "$R03E_RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "$evaluation/crysllmgen_metrics" \
    --eval-model-name mp20

  local cache="$RUN_ROOT/mp_cache/completed_mp_hull_cache.jsonl"
  local cache_sha
  cache_sha="$("$PYTHON" - "$RUN_ROOT/mp_cache/completion_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["completed_mp_hull_cache"]["sha256"])
PY
)"
  test "$cache_sha" = "$(sha256sum "$cache" | cut -d' ' -f1)"

  echo "STAGE arm=$ARM repeat=$REPEAT stage=$stage complete_cache_sun"
  "$PYTHON" "$SOURCE/run_crysllmgen_a100_sun_completed_cache.py" \
    --runner "$R03E_RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
    --runner-sha256 e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de \
    --a100-sun-module-sha256 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968 \
    --expected-mp-hull-cache-sha256 "$cache_sha" \
    -- \
    --generation-jsonl "$generation/generation.jsonl" \
    --expected-attempts 1000 \
    --eval-sun-py reference/a100_eval_sun/eval_sun.py \
    --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
    --mp-hull-cache "$cache" \
    --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
    --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
    --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
    --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
    --execution-patch-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
    --output-dir "$evaluation/r5c_a100_sun"

  echo "STAGE arm=$ARM repeat=$REPEAT stage=$stage validate"
  "$PYTHON" "$SOURCE/validate_stage1000.py" \
    --arm "$ARM" \
    --repeat "$REPEAT" \
    --stage "$stage" \
    --config "$SOURCE/CONFIG.json" \
    --source-dir "$SOURCE" \
    --source-manifest-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
    --generation-dir "$generation" \
    --evaluation-dir "$evaluation" \
    --completion-manifest "$RUN_ROOT/mp_cache/completion_manifest.json"
}

echo "STAGE arm=$ARM repeat=$REPEAT finalize_pre_model494"
export PYTHONPATH="$SOURCE:$R03D/runtime:$R03D"
"$PYTHON" "$SOURCE/finalize_pre1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
  --body-dir "$REPEAT_ROOT/body" \
  --output-dir "$REPEAT_ROOT/stages/pre_model494/generation"
evaluate_stage pre_model494

echo "STAGE arm=$ARM repeat=$REPEAT frozen_model494_refine800"
export PYTHONPATH="$SOURCE:$R03E_RUNTIME:$R03E"
"$PYTHON" "$SOURCE/refine1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --body-attempts "$REPEAT_ROOT/body/body_attempts.jsonl" \
  --proposal-graphs "$REPEAT_ROOT/body/proposal_graphs.pt" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$REPEAT_ROOT/refinement" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen"

echo "STAGE arm=$ARM repeat=$REPEAT finalize_post_model494"
"$PYTHON" "$SOURCE/finalize_post1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
  --refinement-dir "$REPEAT_ROOT/refinement" \
  --output-dir "$REPEAT_ROOT/stages/post_model494/generation"
evaluate_stage post_model494

echo "STAGE arm=$ARM repeat=$REPEAT validate_complete_repeat"
"$PYTHON" "$SOURCE/validate_repeat1000.py" \
  --arm "$ARM" \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
  --repeat-root "$REPEAT_ROOT"

echo "STAGE arm=$ARM repeat=$REPEAT complete"

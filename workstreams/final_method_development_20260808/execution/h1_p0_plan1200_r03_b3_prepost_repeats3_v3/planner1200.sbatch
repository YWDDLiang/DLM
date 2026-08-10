#!/bin/bash
#SBATCH --job-name=h1-p0-plan1200-r3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/planner-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/planner-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/planner_source"
EXECUTION="$SOURCE/workstreams/final_method_development_20260808/execution/h1_p0_plan1200_r03_b3_prepost_repeats3_v3"
MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
CHECKPOINT="$PROJECT/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_PLAN1200_SOURCE_SHA256:?planner source manifest SHA is required}"

case "${SLURM_ARRAY_TASK_ID:?array task required}" in
  0) REPEAT=0; SEED=17029 ;;
  1) REPEAT=1; SEED=27183 ;;
  2) REPEAT=2; SEED=31415 ;;
  *) echo "invalid planner repeat" >&2; exit 2 ;;
esac

REPEAT_ROOT="$RUN_ROOT/repeats/$REPEAT"
PLANNER_DIR="$REPEAT_ROOT/planner1200"
COHORT_DIR="$REPEAT_ROOT/cohort"
EXIT_CODE="$RUN_ROOT/status/planner_repeat_${REPEAT}_exit_code.txt"
test ! -e "$EXIT_CODE"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then
    touch "$RUN_ROOT/status/planner_repeat_${REPEAT}_FAILED"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_PLAN1200_SOURCE_SHA256" ]]; then
  echo "planner resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/preparation_SUCCESS"
test -d "$MODEL"
test -d "$CHECKPOINT"
test "$(sha256sum "$CHECKPOINT/adapter_model.safetensors" | cut -d' ' -f1)" = 65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
test ! -e "$REPEAT_ROOT"
mkdir -p "$REPEAT_ROOT"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/planner_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected exactly one A800, observed {names}")
smact_version = metadata.version("smact")
if not smact_version.startswith("3.1"):
    raise SystemExit(f"SMACT 3.1 required, observed {smact_version}")
print({"gpu": names, "torch": torch.__version__, "smact": smact_version})
PY

echo "STAGE repeat=$REPEAT planner_sample1200 seed=$SEED"
"$PYTHON" "$SOURCE/scripts/sample_llama_h1_formula_plans.py" \
  --model-path "$MODEL" \
  --checkpoint-path "$CHECKPOINT" \
  --output-dir "$PLANNER_DIR" \
  --num-samples 1200 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --seed "$SEED" \
  --seed-mode stateless_ordinal_v1 \
  --formula-constraint-mode off

echo "STAGE repeat=$REPEAT freeze_shared_cohort1000"
"$PYTHON" "$EXECUTION/freeze_planner_cohort.py" \
  --repeat "$REPEAT" \
  --expected-seed "$SEED" \
  --planner-dir "$PLANNER_DIR" \
  --output-dir "$COHORT_DIR"

touch "$RUN_ROOT/status/planner_repeat_${REPEAT}_SUCCESS"
echo "STAGE repeat=$REPEAT planner_complete"

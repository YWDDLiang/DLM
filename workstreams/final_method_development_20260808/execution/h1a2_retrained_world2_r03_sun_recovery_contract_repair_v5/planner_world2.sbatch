#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-w2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/planner-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/planner-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
SOURCE="$RUN_ROOT/source"
SAMPLER="$PROJECT/scripts/sample_llama_h1_formula_plans.py"
MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/
TRAIN_RUN="$PROJECT/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

case "${SLURM_ARRAY_TASK_ID:?array task required}" in
  0) INDEX=0; COHORT=retrained_seed52021_world2_b4; SEED=52021 ;;
  1) INDEX=1; COHORT=retrained_seed62023_world2_b4; SEED=62023 ;;
  2) INDEX=2; COHORT=retrained_seed72031_world2_b4; SEED=72031 ;;
  3) INDEX=3; COHORT=retrained_seed82037_world2_b4; SEED=82037 ;;
  4) INDEX=4; COHORT=retrained_seed17_world2_b4_topology_match; SEED=17 ;;
  *) echo "invalid planner task" >&2; exit 2 ;;
esac

ROOT="$RUN_ROOT/planner/$COHORT"
PLANNER="$ROOT/raw"
FROZEN="$ROOT/frozen"
EXIT_CODE="$RUN_ROOT/status/planner_${INDEX}_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/planner_${INDEX}_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 16
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$SAMPLER" | cut -d' ' -f1)" = d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8
test "$(sha256sum "$PROJECT/crystal_dlm/h1_llm_planner.py" | cut -d' ' -f1)" = d45ccc23fad4284fdeef53d7bbdc5e4044fb6b598092461663473ed8f5a4f8ad
test "$(sha256sum "$PROJECT/crystal_dlm/r5_plan_state.py" | cut -d' ' -f1)" = 6bc1e446afdbe405df601f609a2326c9f376a7f036e9d27dafcecd57965866fc
test "$(sha256sum "$PROJECT/crystal_dlm/r5_plan_body.py" | cut -d' ' -f1)" = 3478ddf657873ea055e5816c423ce36be5ecf0cd1a73c6ee1e5514648047be83
test -f "$TRAIN_RUN/status/training_SUCCESS"
test -f "$TRAIN_RUN/training_terminal_report.json"
test -f "$RUN_ROOT/status/preparation_SUCCESS"
test ! -e "$ROOT"

mapfile -t TRAIN_VALUES < <("$PYTHON" - "$SOURCE/CONFIG.json" "$TRAIN_RUN/training_terminal_report.json" <<'PY'
import json, re, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
terminal = json.load(open(sys.argv[2], encoding="utf-8"))
upstream = config["training_upstream"]
for key in ("adapter_sha256", "adapter_config_sha256"):
    if re.fullmatch(r"[0-9a-f]{64}", str(upstream[key])) is None:
        raise SystemExit(f"unfrozen {key}")
if terminal.get("engineering_status") != "complete" or terminal.get("adapter_sha256") != upstream["adapter_sha256"] or terminal.get("adapter_config_sha256") != upstream["adapter_config_sha256"]:
    raise SystemExit("training terminal/config identity mismatch")
if terminal.get("adapter_path") != upstream["adapter_path"] + "/adapter_model.safetensors":
    raise SystemExit("training terminal adapter path mismatch")
print(upstream["adapter_path"])
print(upstream["adapter_sha256"])
print(upstream["adapter_config_sha256"])
PY
)
CHECKPOINT="${TRAIN_VALUES[0]}"
EXPECTED_ADAPTER="${TRAIN_VALUES[1]}"
EXPECTED_CONFIG="${TRAIN_VALUES[2]}"
test "$(sha256sum "$CHECKPOINT/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$CHECKPOINT/adapter_config.json" | cut -d' ' -f1)" = "$EXPECTED_CONFIG"
mkdir -p "$ROOT"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE:$PROJECT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/planner_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY

"$PYTHON" - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 2 or any("A800" not in name for name in names):
    raise SystemExit(f"expected two A800s, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

echo "STAGE cohort=$COHORT exact_h1a2_world2_planner"
torchrun --nproc_per_node=2 --master_port="$((29000 + (SLURM_ARRAY_JOB_ID + 173 * SLURM_ARRAY_TASK_ID) % 12000))" "$SAMPLER" \
  --model-path "$MODEL" \
  --checkpoint-path "$CHECKPOINT" \
  --output-dir "$PLANNER" \
  --num-samples 1200 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --do-sample \
  --stop-after-plan-marker \
  --truncate-after-plan-marker \
  --seed "$SEED"

"$PYTHON" "$SOURCE/freeze_cohort.py" \
  --cohort-id "$COHORT" \
  --seed "$SEED" \
  --planner-dir "$PLANNER" \
  --output-dir "$FROZEN" \
  --config "$SOURCE/CONFIG.json" \
  --checkpoint-sha256 "$EXPECTED_ADAPTER" \
  --checkpoint-config-sha256 "$EXPECTED_CONFIG"
touch "$RUN_ROOT/status/planner_${INDEX}_SUCCESS"
echo "STAGE cohort=$COHORT planner_complete"

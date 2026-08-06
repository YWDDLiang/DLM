#!/bin/bash
#SBATCH --job-name=v3-planner512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_planner512_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_planner512_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_v3_planner512_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_planner512_v1"
AUTHORIZATION="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/AUTHORIZATION_V3_PLANNER512_DEPENDENCY_SCREEN_20260801.json"
PROTOCOL="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/PROTOCOL_ADDENDUM_V3_PLANNER512_DEPENDENCY_MARGIN_20260801.json"
MANIFEST="${EXECUTION_DIR}/EXECUTION_MANIFEST.json"
TRAINING_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1"
TRAINING_TERMINAL="${TRAINING_ROOT}/terminal_report.json"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
TEACHER_JSONL="${PROJECT_ROOT}/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base/train.jsonl"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_AUTH_SHA256:?submission must export EXPECTED_AUTH_SHA256}"
: "${EXPECTED_PROTOCOL_SHA256:?submission must export EXPECTED_PROTOCOL_SHA256}"
: "${EXPECTED_MANIFEST_SHA256:?submission must export EXPECTED_MANIFEST_SHA256}"

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 2 ]; then
  echo "registered Planner-512 Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=P0
    STEP=2
    CHECKPOINT="${P0_CHECKPOINT}"
    CHECKPOINT_IDENTITY=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
    ;;
  1)
    ARM=P-control
    STEP=400
    CHECKPOINT="${TRAINING_ROOT}/arms/pcontrol/checkpoint-0400/adapter"
    CHECKPOINT_IDENTITY=8fa09305d22d113e7567daefaba886fe0774feb9ece7c13664214670d20d6c50
    ;;
  2)
    ARM=P-star
    STEP=400
    CHECKPOINT="${TRAINING_ROOT}/arms/pstar/checkpoint-0400/adapter"
    CHECKPOINT_IDENTITY=617aebf19e9b13ec84d7751bdc3824c13eb4fb639f8ccc5067dcf5fc3f84e950
    ;;
  *)
    exit 2
    ;;
esac

ARM_ROOT="${RUN_ROOT}/arms/${ARM}"
PLAN_ROOT="${ARM_ROOT}/plan512"

test -d "${SOURCE_ROOT}"
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test -f "${TEACHER_JSONL}"
test -f "${TRAINING_TERMINAL}"
test ! -e "${ARM_ROOT}"
test "$(sha256sum "${AUTHORIZATION}" | cut -d' ' -f1)" = "${EXPECTED_AUTH_SHA256}"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = "${EXPECTED_PROTOCOL_SHA256}"
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
if [ "${ARM}" = P0 ]; then
  test "$(sha256sum "${CHECKPOINT}/adapter_model.safetensors" | cut -d' ' -f1)" = "${CHECKPOINT_IDENTITY}"
else
  test "$(sha256sum "$(dirname "${CHECKPOINT}")/checkpoint_manifest.json" | cut -d' ' -f1)" = "${CHECKPOINT_IDENTITY}"
fi

cd "${SOURCE_ROOT}"
sha256sum -c "${EXECUTION_DIR}/SOURCE_SHA256.txt"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

"${PYTHON}" - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

mkdir -p "${ARM_ROOT}"
"${PYTHON}" scripts/sample_llama_h1_formula_plans.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${PLAN_ROOT}" \
  --num-samples 512 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --seed 17029 \
  --seed-mode stateless_ordinal_v1

"${PYTHON}" "${EXECUTION_DIR}/evaluate_planner_arm.py" \
  --arm "${ARM}" \
  --step "${STEP}" \
  --raw-generations "${PLAN_ROOT}/raw_generations.jsonl" \
  --sample-metrics "${PLAN_ROOT}/sample_metrics.json" \
  --run-config "${PLAN_ROOT}/run_config.json" \
  --teacher-jsonl "${TEACHER_JSONL}" \
  --execution-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --checkpoint-identity-sha256 "${CHECKPOINT_IDENTITY}" \
  --output "${ARM_ROOT}/plan_report.json"

echo "PLANNER512_ARM_COMPLETE ARM=${ARM} STEP=${STEP}"

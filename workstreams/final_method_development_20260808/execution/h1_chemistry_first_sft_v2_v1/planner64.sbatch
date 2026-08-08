#!/bin/bash
#SBATCH --job-name=h1-cf-p64-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3/logs/%A_%a_planner64.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3/logs/%A_%a_planner64.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
STAGE=64
BASE_SEED=26080964

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${EXPECTED_LEDGER_SHA256:?missing planner64 ledger identity}"
: "${LEGACY_PYTHON:?missing generation runtime}"
: "${SLURM_ARRAY_TASK_ID:?missing arm identity}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=p0; PROMPT_STYLE=h1_rich_plan_v1; CHECKPOINT="${P0_ADAPTER}" ;;
  1) ARM=sft_v2; PROMPT_STYLE=h1_rich_nocharge_plan_v1 ;;
  2) ARM=sft_v2_c; PROMPT_STYLE=h1_rich_nocharge_plan_v1 ;;
  *) exit 3 ;;
esac
test "${SLURM_JOB_PARTITION:-}" = gpu
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${EXECUTION_DIR}/LEDGER64.json" | cut -d' ' -f1)" = "${EXPECTED_LEDGER_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

if test "${ARM}" != p0; then
  if test -f "${RUN_ROOT}/status/train_${ARM}_SCIENTIFIC_STOP"; then
    touch "${RUN_ROOT}/status/planner64_${ARM}_SKIPPED_TRAINING_SCIENTIFIC_STOP"
    echo "H1_CHEMISTRY_FIRST_PLANNER64_SKIP candidate=${ARM}"
    exit 0
  fi
  test -f "${RUN_ROOT}/status/train_${ARM}_SUCCESS"
  CHECKPOINT="$("${LEGACY_PYTHON}" "${EXECUTION_DIR}/resolve_fixed_adapter.py" \
    --training-dir "${RUN_ROOT}/training/${ARM}" --candidate "${ARM}")"
fi
test ! -e "${RUN_ROOT}/planner64/${ARM}"

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/planner64" "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/planner64_${ARM}_exit_code.txt"
  touch "${RUN_ROOT}/status/planner64_${ARM}_FAILED"
  exit "${rc}"
}
trap on_error ERR

"${LEGACY_PYTHON}" scripts/sample_llama_h1_formula_plans.py \
  --model-path "${MODEL_PATH}" --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner64/${ARM}" \
  --num-samples "${STAGE}" --batch-size 1 --max-new-tokens 96 \
  --temperature 0.9 --top-p 0.95 --top-k 50 --max-atoms 20 \
  --prompt-style "${PROMPT_STYLE}" --no-include-sample-id \
  --seed "${BASE_SEED}" --seed-mode stateless_ordinal_v1 \
  --formula-constraint-mode off --formula-missing-state-policy allow_non_applicable
test "$(wc -l < "${RUN_ROOT}/planner64/${ARM}/raw_generations.jsonl")" -eq 64
printf '0\n' > "${RUN_ROOT}/status/planner64_${ARM}_exit_code.txt"
touch "${RUN_ROOT}/status/planner64_${ARM}_SUCCESS"
echo "H1_CHEMISTRY_FIRST_PLANNER64_ARM_PASS arm=${ARM}"

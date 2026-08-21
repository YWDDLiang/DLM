#!/usr/bin/env bash
#SBATCH --job-name=h1a2-plans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-plans-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly CHECKPOINT="${REPO_ROOT}/checkpoints/planner"
readonly BASE_MODEL="${REPO_ROOT}/checkpoints/planner/base"
readonly MODE="${PLAN_SAMPLE_MODE:-full}"

if [[ "${MODE}" == "quick" ]]; then
  NUM_SAMPLES=256
  SEED=17029
  WORLD_SIZE=1
  OUTPUT="${REPO_ROOT}/runs/quick_256x4/plans"
else
  NUM_SAMPLES=1200
  SEED=17
  WORLD_SIZE=2
  OUTPUT="${REPO_ROOT}/runs/h1a2_full/plans"
fi

[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]] || missing "${CHECKPOINT}/adapter_model.safetensors"
[[ -d "${BASE_MODEL}" ]] || missing "${BASE_MODEL}"
mkdir -p "${OUTPUT}"

torchrun --nproc_per_node="${WORLD_SIZE}" "${REPO_ROOT}/src/scripts/sample_llama_h1_formula_plans.py" \
  --model-path "${BASE_MODEL}" --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" --num-samples "${NUM_SAMPLES}" --batch-size 4 \
  --max-new-tokens 96 --temperature 0.9 --top-p 0.95 --top-k 50 \
  --max-atoms 20 --prompt-style h1_rich_plan_v1 --no-include-sample-id \
  --seed "${SEED}" --do-sample --stop-after-plan-marker \
  --truncate-after-plan-marker


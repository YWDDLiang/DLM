#!/usr/bin/env bash
#SBATCH --job-name=h1a2-difficulty-sample
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=h1a2-difficulty-sample-%j.out

set -Eeuo pipefail
readonly CANDIDATE_ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT before sbatch}"
source "${CANDIDATE_ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_PLANNER_MODEL_PATH:-${REPO_ROOT}/checkpoints/planner/base}"
readonly TRAIN_RUN="${H1A2_TRAIN_RUN:?export H1A2_TRAIN_RUN with the completed four-arm training root}"
readonly RUN_DIR="${REPO_ROOT}/runs/difficulty_planner_sample_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

sample_arm() {
  local device="$1"
  local checkpoint="$2"
  local output="$3"
  local seed="$4"
  CUDA_VISIBLE_DEVICES="${device}" python "${REPO_ROOT}/src/scripts/sample_llama_h1_formula_plans.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --num-samples 256 \
    --batch-size 4 \
    --max-new-tokens 96 \
    --temperature 0.9 \
    --top-p 0.95 \
    --top-k 50 \
    --prompt-style h1_rich_plan_v1 \
    --seed "${seed}"
}

sample_arm 0 "${TRAIN_RUN}/control_seed17/final" "${RUN_DIR}/plans_control_seed17" 17017 & p0=$!
sample_arm 1 "${TRAIN_RUN}/candidate_seed17/final" "${RUN_DIR}/plans_candidate_seed17" 17017 & p1=$!
sample_arm 2 "${TRAIN_RUN}/control_seed18/final" "${RUN_DIR}/plans_control_seed18" 18018 & p2=$!
sample_arm 3 "${TRAIN_RUN}/candidate_seed18/final" "${RUN_DIR}/plans_candidate_seed18" 18018 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"
touch "${RUN_DIR}/_SUCCESS"
log "Four rich-Plan 256 screens complete: ${RUN_DIR}"

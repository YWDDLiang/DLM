#!/usr/bin/env bash
#SBATCH --job-name=h1a2-difficulty-plan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-difficulty-plan-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly MODEL="${H1A2_PLANNER_MODEL_PATH:-${REPO_ROOT}/checkpoints/planner/base}"
readonly P0="${H1A2_PLANNER_P0_PATH:-${REPO_ROOT}/checkpoints/planner}"
readonly ANCHOR="${H1A2_PLANNER_ANCHOR_DATA:-${REPO_ROOT}/data/planner_h1a2}"
readonly ATTEMPTS="${H1A2_SELF_IMPROVEMENT_ATTEMPTS:-${REPO_ROOT}/data/analysis/h1a2_attempts.jsonl}"
readonly RUN_DIR="${REPO_ROOT}/runs/difficulty_planner_${SLURM_JOB_ID}"
readonly WEIGHTED_DATA="${RUN_DIR}/weighted_data"

for path in "${MODEL}" "${P0}" "${ANCHOR}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${ATTEMPTS}" ]] || missing "${ATTEMPTS}"
mkdir -p "${RUN_DIR}"

python "${REPO_ROOT}/scripts/difficulty_program.py" build-buffer \
  --attempts "${ATTEMPTS}" \
  --cohort-id h1a2-self-improvement-v1 \
  --anchor-dir "${ANCHOR}" \
  --output-dir "${WEIGHTED_DATA}" \
  --buffer-fraction 0.05 \
  --folds 5 \
  --prior-strength 20 \
  --alpha 1 \
  --beta 1 \
  --max-weight 5 \
  --min-ess-ratio 0.5

train_arm() {
  local device="$1"
  local data="$2"
  local output="$3"
  local seed="$4"
  local difficulty="$5"
  local extra=()
  if [[ "${difficulty}" == true ]]; then
    extra+=(--difficulty-decomposed-weighting --difficulty-manifest "${WEIGHTED_DATA}/difficulty_manifest.json")
  fi
  CUDA_VISIBLE_DEVICES="${device}" python "${REPO_ROOT}/src/scripts/llama_formula_sft.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${P0}" \
    --data-dir "${data}" \
    --output-dir "${output}" \
    --max-length 768 \
    --epochs 1 \
    --max-updates 400 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-steps 25 \
    --eval-steps 100 \
    --seed "${seed}" \
    "${extra[@]}"
}

train_arm 0 "${ANCHOR}" "${RUN_DIR}/control_seed17" 17 false & p0=$!
train_arm 1 "${WEIGHTED_DATA}" "${RUN_DIR}/candidate_seed17" 17 true & p1=$!
train_arm 2 "${ANCHOR}" "${RUN_DIR}/control_seed18" 18 false & p2=$!
train_arm 3 "${WEIGHTED_DATA}" "${RUN_DIR}/candidate_seed18" 18 true & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

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
    --seed "${seed}"
}

sample_arm 0 "${RUN_DIR}/control_seed17/final" "${RUN_DIR}/plans_control_seed17" 17017 & p0=$!
sample_arm 1 "${RUN_DIR}/candidate_seed17/final" "${RUN_DIR}/plans_candidate_seed17" 17017 & p1=$!
sample_arm 2 "${RUN_DIR}/control_seed18/final" "${RUN_DIR}/plans_control_seed18" 18018 & p2=$!
sample_arm 3 "${RUN_DIR}/candidate_seed18/final" "${RUN_DIR}/plans_candidate_seed18" 18018 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

touch "${RUN_DIR}/_SUCCESS"
log "Difficulty Planner control/candidate training and Plan-256 screens complete: ${RUN_DIR}"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-plan-meta-v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-planner-meta-guard-v4-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_PLANNER_MODEL_PATH:?export H1A2_PLANNER_MODEL_PATH}"
readonly P0="${H1A2_PLANNER_P0_PATH:?export H1A2_PLANNER_P0_PATH}"
readonly ANCHOR="${H1A2_PLANNER_ANCHOR_DATA:?export H1A2_PLANNER_ANCHOR_DATA}"
readonly ATTEMPTS="${H1A2_SELF_IMPROVEMENT_ATTEMPTS:?export H1A2_SELF_IMPROVEMENT_ATTEMPTS}"
readonly RUN_DIR="${REPO_ROOT}/runs/difficulty_planner_meta_guard_v4_${SLURM_JOB_ID}"
readonly WEIGHTED_DATA="${RUN_DIR}/weighted_data"

for path in "${MODEL}" "${P0}" "${ANCHOR}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${ATTEMPTS}" ]] || missing "${ATTEMPTS}"
mkdir -p "${RUN_DIR}"

python "${REPO_ROOT}/scripts/difficulty_program.py" build-buffer \
  --attempts "${ATTEMPTS}" \
  --cohort-id h1a2-meta-guard-v4 \
  --anchor-dir "${ANCHOR}" \
  --output-dir "${WEIGHTED_DATA}" \
  --buffer-fraction 0.20 \
  --folds 5 \
  --prior-strength 20 \
  --alpha 0 \
  --beta 1 \
  --meta-reward-weight 2 \
  --strict-reward-weight 1 \
  --max-weight 5 \
  --min-ess-ratio 0.5

train_candidate() {
  local device="$1" output="$2" seed="$3"
  CUDA_VISIBLE_DEVICES="${device}" python "${REPO_ROOT}/src/scripts/llama_formula_sft.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${P0}" \
    --data-dir "${WEIGHTED_DATA}" \
    --output-dir "${output}" \
    --max-length 768 \
    --epochs 1 \
    --max-updates 400 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-steps 25 \
    --eval-steps 100 \
    --logging-steps 20 \
    --seed "${seed}" \
    --difficulty-decomposed-weighting \
    --difficulty-manifest "${WEIGHTED_DATA}/difficulty_manifest.json" \
    --weighted-sampling \
    --sampling-weight-key difficulty_sampling_weight
}

train_candidate 0 "${RUN_DIR}/candidate_seed17" 17 & p0=$!
train_candidate 1 "${RUN_DIR}/candidate_seed18" 18 & p1=$!
wait "${p0}"; wait "${p1}"

sample_arm() {
  local device="$1" checkpoint="$2" output="$3" seed="$4"
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

sample_arm 0 "${P0}" "${RUN_DIR}/plans_control_seed17" 17017 & p0=$!
sample_arm 1 "${RUN_DIR}/candidate_seed17/final" "${RUN_DIR}/plans_candidate_seed17" 17017 & p1=$!
sample_arm 2 "${P0}" "${RUN_DIR}/plans_control_seed18" 18018 & p2=$!
sample_arm 3 "${RUN_DIR}/candidate_seed18/final" "${RUN_DIR}/plans_candidate_seed18" 18018 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

touch "${RUN_DIR}/_SUCCESS"
log "Meta-guard Planner V4 and original-P0 Plan-256 controls complete: ${RUN_DIR}"

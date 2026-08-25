#!/usr/bin/env bash
#SBATCH --job-name=h1a2-grounding
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-grounding-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:-${REPO_ROOT}/checkpoints/dlm/base}"
readonly B0="${H1A2_DLM_B0_PATH:-${REPO_ROOT}/checkpoints/dlm}"
readonly DATA="${H1A2_DLM_DATA_PATH:-${REPO_ROOT}/data/dlm_r5_exact_length}"
readonly PLANS="${H1A2_SCREEN_PLANS:-${REPO_ROOT}/data/plans/h1a2_parsed_1186.jsonl}"
readonly REFINER="${H1A2_REFINER_PATH:-${REPO_ROOT}/checkpoints/diffusion/model_494.pt}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:-${REPO_ROOT}/src/vendor/crysllmgen}"
readonly WEIGHT="${H1A2_GROUNDING_WEIGHT:-0.1}"
readonly RUN_DIR="${REPO_ROOT}/runs/grounding_${SLURM_JOB_ID}"
readonly GROUNDED_DATA="${RUN_DIR}/data_grounded"

for path in "${MODEL}" "${B0}" "${DATA}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${PLANS}" ]] || missing "${PLANS}"
[[ -f "${REFINER}" ]] || missing "${REFINER}"
mkdir -p "${RUN_DIR}"

python "${REPO_ROOT}/scripts/build_counterfactual_grounding_data.py" \
  --input-dir "${DATA}" \
  --output-dir "${GROUNDED_DATA}" \
  --seed 17

train_arm() {
  local devices="$1"
  local port="$2"
  local data="$3"
  local output="$4"
  local grounding_weight="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REPO_ROOT}/src/scripts/llada_sft.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${B0}" \
    --data-dir "${data}" \
    --output-dir "${output}" \
    --representation dynamic_v1 \
    --max-length 382 \
    --answer-token-count 87 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 5e-5 \
    --save-steps 1696 \
    --eval-steps 500 \
    --seed 17017 \
    --counterfactual-grounding-weight "${grounding_weight}"
}

train_arm 0,1 29631 "${DATA}" "${RUN_DIR}/control" 0.0 &
control_pid=$!
train_arm 2,3 29632 "${GROUNDED_DATA}" "${RUN_DIR}/candidate" "${WEIGHT}" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

sample_arm() {
  local devices="$1"
  local port="$2"
  local checkpoint="$3"
  local output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REPO_ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 256 \
    --batch-size 1 \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

sample_arm 0,1 29633 "${RUN_DIR}/control/final" "${RUN_DIR}/screen_control/body" &
control_pid=$!
sample_arm 2,3 29634 "${RUN_DIR}/candidate/final" "${RUN_DIR}/screen_candidate/body" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

refine_arm() {
  local devices="$1"
  local port="$2"
  local proposals="$3"
  local output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REPO_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${proposals}" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --diff-steps 800 \
    --max-proposals 256
}

refine_arm 0,1 29635 "${RUN_DIR}/screen_control/body/proposal_graphs.pt" "${RUN_DIR}/screen_control/refine" &
control_pid=$!
refine_arm 2,3 29636 "${RUN_DIR}/screen_candidate/body/proposal_graphs.pt" "${RUN_DIR}/screen_candidate/refine" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

touch "${RUN_DIR}/_SUCCESS"
log "Grounding control/candidate training and fixed-256 body/refine complete: ${RUN_DIR}"


#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-screen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-ground-screen-%j.out

set -Eeuo pipefail
readonly CANDIDATE_ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT before sbatch}"
readonly PARENT_RUN="${H1A2_PARENT_RUN:?export H1A2_PARENT_RUN with the completed control/candidate checkpoints}"
source "${CANDIDATE_ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:-${REPO_ROOT}/checkpoints/dlm/base}"
readonly PLANS="${H1A2_SCREEN_PLANS:-${REPO_ROOT}/data/plans/h1a2_parsed_1186.jsonl}"
readonly REFINER="${H1A2_REFINER_PATH:-${REPO_ROOT}/checkpoints/diffusion/model_494.pt}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:-${REPO_ROOT}/src/vendor/crysllmgen}"
readonly DLM_SEED="${H1A2_DLM_SEED:-17017}"
readonly REFINER_SEED="${H1A2_REFINER_SEED:-27017}"
readonly RUN_DIR="${REPO_ROOT}/runs/grounding_screen_resume_${SLURM_JOB_ID}"

for path in "${MODEL}" "${CRYS}" "${PARENT_RUN}/control/final" "${PARENT_RUN}/candidate/final"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${PLANS}" ]] || missing "${PLANS}"
[[ -f "${REFINER}" ]] || missing "${REFINER}"
mkdir -p "${RUN_DIR}"

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
    --seed "${DLM_SEED}" \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

sample_arm 0,1 29733 "${PARENT_RUN}/control/final" "${RUN_DIR}/screen_control/body" &
control_pid=$!
sample_arm 2,3 29734 "${PARENT_RUN}/candidate/final" "${RUN_DIR}/screen_candidate/body" &
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
    --seed "${REFINER_SEED}" \
    --diff-steps 800 \
    --max-proposals 256
}

refine_arm 0,1 29735 "${RUN_DIR}/screen_control/body/proposal_graphs.pt" "${RUN_DIR}/screen_control/refine" &
control_pid=$!
refine_arm 2,3 29736 "${RUN_DIR}/screen_candidate/body/proposal_graphs.pt" "${RUN_DIR}/screen_candidate/refine" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

touch "${RUN_DIR}/_SUCCESS"
log "Grounding fixed-256 resume complete from ${PARENT_RUN}: ${RUN_DIR}"

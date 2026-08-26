#!/usr/bin/env bash
#SBATCH --job-name=h1a2-meta-v4-d4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-meta-v4-downstream4-%j.out

set -Eeuo pipefail
readonly CANDIDATE_ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLAN_RUN="${H1A2_PLAN_RUN:?export H1A2_PLAN_RUN}"
readonly GROUNDING_ROOT="${H1A2_GROUNDING_ROOT:?export H1A2_GROUNDING_ROOT}"
source "${CANDIDATE_ROOT}/slurm/_common.sh"
activate_environment
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly DLM_CHECKPOINT="${H1A2_DLM_CHECKPOINT:?export H1A2_DLM_CHECKPOINT}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${GROUNDING_ROOT}/src/vendor/crysllmgen"
readonly SAMPLE_SCRIPT="${GROUNDING_ROOT}/src/scripts/sample_llada_r5_exact_length.py"
readonly REFINE_SCRIPT="${GROUNDING_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly RUN_DIR="${REPO_ROOT}/runs/difficulty_v4_downstream_${SLURM_JOB_ID}"
for path in "${MODEL}" "${DLM_CHECKPOINT}" "${REFINER}" "${CRYS}" "${SAMPLE_SCRIPT}" "${REFINE_SCRIPT}" "${PLAN_RUN}/_SUCCESS"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${RUN_DIR}"

sample_cell() {
  local devices="$1" port="$2" plans="$3" output="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${SAMPLE_SCRIPT}" --model-path "${MODEL}" --checkpoint-path "${DLM_CHECKPOINT}" \
    --output-dir "${output}" --crysllmgen-dir "${CRYS}" --prompt-jsonl "${plans}" \
    --num-samples 256 --batch-size 1 --no-repeat-prompt-records \
    --preserve-prompt-sample-idx --seed "${seed}" --seed-by-sample-index \
    --temperature 0.7 --generation-schedule exact-plan
}

sample_pair() {
  local seed="$1" base_port="$2" dlm_seed="$3"
  sample_cell 0,1 "${base_port}1" "${PLAN_RUN}/plans_control_seed${seed}/plans_for_dlm.jsonl" "${RUN_DIR}/seed${seed}/control/body" "${dlm_seed}" & p0=$!
  sample_cell 2,3 "${base_port}2" "${PLAN_RUN}/plans_candidate_seed${seed}/plans_for_dlm.jsonl" "${RUN_DIR}/seed${seed}/candidate/body" "${dlm_seed}" & p1=$!
  wait "${p0}"; wait "${p1}"
}
sample_pair 17 3060 17117
sample_pair 18 3061 18118

refine_cell() {
  local devices="$1" port="$2" proposals="$3" output="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REFINE_SCRIPT}" --proposal-graphs "${proposals}" --checkpoint "${REFINER}" \
    --output-dir "${output}" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${seed}" --seed-by-sample-index --diff-steps 800 --max-proposals 256
}

refine_pair() {
  local seed="$1" base_port="$2" refiner_seed="$3"
  refine_cell 0,1 "${base_port}1" "${RUN_DIR}/seed${seed}/control/body/proposal_graphs.pt" "${RUN_DIR}/seed${seed}/control/refine" "${refiner_seed}" & p0=$!
  refine_cell 2,3 "${base_port}2" "${RUN_DIR}/seed${seed}/candidate/body/proposal_graphs.pt" "${RUN_DIR}/seed${seed}/candidate/refine" "${refiner_seed}" & p1=$!
  wait "${p0}"; wait "${p1}"
}
refine_pair 17 3070 27117
refine_pair 18 3071 28118
touch "${RUN_DIR}/_SUCCESS"
log "Meta-guard Planner V4 four-cell body/refine complete on 4 GPUs: ${RUN_DIR}"

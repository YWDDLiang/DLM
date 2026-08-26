#!/usr/bin/env bash
#SBATCH --job-name=h1a2-meta-v4-down
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=720G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-meta-v4-downstream-%j.out

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
    "${SAMPLE_SCRIPT}" \
    --model-path "${MODEL}" --checkpoint-path "${DLM_CHECKPOINT}" \
    --output-dir "${output}" --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${plans}" --num-samples 256 --batch-size 1 \
    --no-repeat-prompt-records --preserve-prompt-sample-idx \
    --seed "${seed}" --seed-by-sample-index --temperature 0.7 \
    --generation-schedule exact-plan
}

sample_cell 0,1 30401 "${PLAN_RUN}/plans_control_seed17/plans_for_dlm.jsonl" "${RUN_DIR}/seed17/control/body" 17117 & p0=$!
sample_cell 2,3 30402 "${PLAN_RUN}/plans_candidate_seed17/plans_for_dlm.jsonl" "${RUN_DIR}/seed17/candidate/body" 17117 & p1=$!
sample_cell 4,5 30403 "${PLAN_RUN}/plans_control_seed18/plans_for_dlm.jsonl" "${RUN_DIR}/seed18/control/body" 18118 & p2=$!
sample_cell 6,7 30404 "${PLAN_RUN}/plans_candidate_seed18/plans_for_dlm.jsonl" "${RUN_DIR}/seed18/candidate/body" 18118 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

refine_cell() {
  local devices="$1" port="$2" proposals="$3" output="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REFINE_SCRIPT}" --proposal-graphs "${proposals}" --checkpoint "${REFINER}" \
    --output-dir "${output}" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${seed}" --seed-by-sample-index --diff-steps 800 --max-proposals 256
}

refine_cell 0,1 30501 "${RUN_DIR}/seed17/control/body/proposal_graphs.pt" "${RUN_DIR}/seed17/control/refine" 27117 & p0=$!
refine_cell 2,3 30502 "${RUN_DIR}/seed17/candidate/body/proposal_graphs.pt" "${RUN_DIR}/seed17/candidate/refine" 27117 & p1=$!
refine_cell 4,5 30503 "${RUN_DIR}/seed18/control/body/proposal_graphs.pt" "${RUN_DIR}/seed18/control/refine" 28118 & p2=$!
refine_cell 6,7 30504 "${RUN_DIR}/seed18/candidate/body/proposal_graphs.pt" "${RUN_DIR}/seed18/candidate/refine" 28118 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

touch "${RUN_DIR}/_SUCCESS"
log "Meta-guard Planner V4 four-cell body/refine complete: ${RUN_DIR}"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-ground-1000-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly FULL_RUN="${H1A2_FULL_RUN:?export H1A2_FULL_RUN}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly PLANS="${H1A2_FIXED1000_PLANS:?export H1A2_FIXED1000_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/grounding_fixed1000_${SLURM_JOB_ID}"
readonly DLM_SEED=18117
readonly REFINER_SEED=28117

for path in "${MODEL}" "${PLANS}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for arm in control candidate; do
  [[ -f "${FULL_RUN}/${arm}/final/adapter_model.safetensors" ]] || \
    missing "${FULL_RUN}/${arm}/final/adapter_model.safetensors"
done
mkdir -p "${RUN_DIR}"
printf 'plan_source\tcohort_rule\trequested\tdlm_seed\trefiner_seed\n%s\tfirst_1000_parsed_plans_no_survivor_filter\t1000\t%d\t%d\n' \
  "${PLANS}" "${DLM_SEED}" "${REFINER_SEED}" >"${RUN_DIR}/scientific_contract.tsv"

sample_arm() {
  local devices="$1" port="$2" checkpoint="$3" output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 1000 \
    --batch-size 1 \
    --seed "${DLM_SEED}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

sample_arm 0,1 30601 "${FULL_RUN}/control/final" "${RUN_DIR}/control/body" &
control_pid=$!
sample_arm 2,3 30602 "${FULL_RUN}/candidate/final" "${RUN_DIR}/candidate/body" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

refine_arm() {
  local devices="$1" port="$2" proposals="$3" output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${proposals}" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed "${REFINER_SEED}" \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 1000
}

refine_arm 0,1 30603 "${RUN_DIR}/control/body/proposal_graphs.pt" "${RUN_DIR}/control/refine" &
control_pid=$!
refine_arm 2,3 30604 "${RUN_DIR}/candidate/body/proposal_graphs.pt" "${RUN_DIR}/candidate/refine" &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

touch "${RUN_DIR}/_SUCCESS"
log "Grounding fixed requested-1000 body/refine complete: ${RUN_DIR}"

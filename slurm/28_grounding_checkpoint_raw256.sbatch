#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-raw256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-ground-raw256-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly TRAIN_RUN="${H1A2_TRAIN_RUN:?export H1A2_TRAIN_RUN}"
readonly FULL_RUN="${H1A2_FULL_RUN:?export H1A2_FULL_RUN}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly PLANS="${H1A2_RAW256_PLANS:?export H1A2_RAW256_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/grounding_checkpoint_raw256_${SLURM_JOB_ID}"
readonly STEPS=(500 1000 1696)
readonly DLM_SEED=17117
readonly REFINER_SEED=27117

for path in "${MODEL}" "${PLANS}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for step in 500 1000; do
  for arm in control candidate; do
    [[ -f "${TRAIN_RUN}/${arm}/checkpoints/step-${step}/adapter_model.safetensors" ]] || \
      missing "${TRAIN_RUN}/${arm}/checkpoints/step-${step}/adapter_model.safetensors"
  done
done
for arm in control candidate; do
  [[ -f "${FULL_RUN}/${arm}/final/adapter_model.safetensors" ]] || \
    missing "${FULL_RUN}/${arm}/final/adapter_model.safetensors"
done
mkdir -p "${RUN_DIR}"
printf 'plan_source\trequested\tdlm_seed\trefiner_seed\n%s\t256\t%d\t%d\n' \
  "${PLANS}" "${DLM_SEED}" "${REFINER_SEED}" >"${RUN_DIR}/scientific_contract.tsv"

checkpoint_for() {
  local step="$1" arm="$2"
  if [[ "${step}" == 1696 ]]; then
    printf '%s\n' "${FULL_RUN}/${arm}/final"
  else
    printf '%s\n' "${TRAIN_RUN}/${arm}/checkpoints/step-${step}"
  fi
}

sample_arm() {
  local devices="$1" port="$2" checkpoint="$3" output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 256 \
    --batch-size 1 \
    --seed "${DLM_SEED}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

for index in 0 1 2; do
  step="${STEPS[$index]}"
  label="$(printf '%04d' "${step}")"
  port_base="$((30401 + index * 2))"
  sample_arm 0,1 "${port_base}" "$(checkpoint_for "${step}" control)" \
    "${RUN_DIR}/step${label}/control/body" &
  control_pid=$!
  sample_arm 2,3 "$((port_base + 1))" "$(checkpoint_for "${step}" candidate)" \
    "${RUN_DIR}/step${label}/candidate/body" &
  candidate_pid=$!
  wait "${control_pid}"
  wait "${candidate_pid}"
done

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
    --max-proposals 256
}

for index in 0 1 2; do
  step="${STEPS[$index]}"
  label="$(printf '%04d' "${step}")"
  port_base="$((30501 + index * 2))"
  refine_arm 0,1 "${port_base}" "${RUN_DIR}/step${label}/control/body/proposal_graphs.pt" \
    "${RUN_DIR}/step${label}/control/refine" &
  control_pid=$!
  refine_arm 2,3 "$((port_base + 1))" "${RUN_DIR}/step${label}/candidate/body/proposal_graphs.pt" \
    "${RUN_DIR}/step${label}/candidate/refine" &
  candidate_pid=$!
  wait "${control_pid}"
  wait "${candidate_pid}"
done

touch "${RUN_DIR}/_SUCCESS"
log "Grounding 500/1000/1696 raw-256 checkpoint generation complete: ${RUN_DIR}"

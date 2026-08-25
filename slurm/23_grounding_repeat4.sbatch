#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-r4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=720G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-ground-repeat4-%j.out

set -Eeuo pipefail
readonly CANDIDATE_ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT before sbatch}"
readonly PARENT_RUN="${H1A2_PARENT_RUN:?export H1A2_PARENT_RUN with completed control/candidate checkpoints}"
source "${CANDIDATE_ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly PLANS="${H1A2_SCREEN_PLANS:?export H1A2_SCREEN_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:-${REPO_ROOT}/src/vendor/crysllmgen}"
readonly RUN_DIR="${REPO_ROOT}/runs/grounding_repeat4_${SLURM_JOB_ID}"
readonly DLM_SEEDS=(17117 17217 17317 17417)
readonly REFINER_SEEDS=(27117 27217 27317 27417)

for path in "${MODEL}" "${CRYS}" "${PARENT_RUN}/control/final" "${PARENT_RUN}/candidate/final"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${PLANS}" ]] || missing "${PLANS}"
[[ -f "${REFINER}" ]] || missing "${REFINER}"
mkdir -p "${RUN_DIR}"
printf 'repeat\tdlm_seed\trefiner_seed\n' > "${RUN_DIR}/repeat_seeds.tsv"
for idx in 0 1 2 3; do
  printf '%d\t%d\t%d\n' "$((idx + 1))" "${DLM_SEEDS[$idx]}" "${REFINER_SEEDS[$idx]}" >> "${RUN_DIR}/repeat_seeds.tsv"
done

sample_arm() {
  local devices="$1" port="$2" checkpoint="$3" output="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REPO_ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 256 \
    --batch-size 1 \
    --seed "${seed}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

run_sample_wave() {
  local first_idx="$1"
  local pids=()
  for offset in 0 1; do
    local idx="$((first_idx + offset))"
    local repeat="$((idx + 1))"
    local base_gpu="$((offset * 4))"
    local base_port="$((29801 + idx * 2))"
    sample_arm "${base_gpu},$((base_gpu + 1))" "${base_port}" \
      "${PARENT_RUN}/control/final" "${RUN_DIR}/repeat$(printf '%02d' "${repeat}")/control/body" "${DLM_SEEDS[$idx]}" &
    pids+=("$!")
    sample_arm "$((base_gpu + 2)),$((base_gpu + 3))" "$((base_port + 1))" \
      "${PARENT_RUN}/candidate/final" "${RUN_DIR}/repeat$(printf '%02d' "${repeat}")/candidate/body" "${DLM_SEEDS[$idx]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

run_sample_wave 0
run_sample_wave 2

refine_arm() {
  local devices="$1" port="$2" proposals="$3" output="$4" seed="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${REPO_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${proposals}" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed "${seed}" \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 256
}

run_refine_wave() {
  local first_idx="$1"
  local pids=()
  for offset in 0 1; do
    local idx="$((first_idx + offset))"
    local repeat="$((idx + 1))"
    local base_gpu="$((offset * 4))"
    local base_port="$((29901 + idx * 2))"
    local repeat_dir="${RUN_DIR}/repeat$(printf '%02d' "${repeat}")"
    refine_arm "${base_gpu},$((base_gpu + 1))" "${base_port}" \
      "${repeat_dir}/control/body/proposal_graphs.pt" "${repeat_dir}/control/refine" "${REFINER_SEEDS[$idx]}" &
    pids+=("$!")
    refine_arm "$((base_gpu + 2)),$((base_gpu + 3))" "$((base_port + 1))" \
      "${repeat_dir}/candidate/body/proposal_graphs.pt" "${repeat_dir}/candidate/refine" "${REFINER_SEEDS[$idx]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

run_refine_wave 0
run_refine_wave 2

touch "${RUN_DIR}/_SUCCESS"
log "Grounding repeat4 body/refine complete: ${RUN_DIR}"

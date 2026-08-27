#!/usr/bin/env bash
#SBATCH --job-name=h1a2-energy-pairs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-energy-pairs-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLANS="${H1A2_ENERGY_PAIR_PLANS:?export H1A2_ENERGY_PAIR_PLANS}"
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_SUFFICIENT_CHECKPOINT:?export H1A2_SUFFICIENT_CHECKPOINT}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly STREAM_OFFSET="${H1A2_STREAM_OFFSET:-0}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly RUN_DIR="${ROOT}/runs/dlm_energy_pair_bodies_${SLURM_JOB_ID}"
[[ "${STREAM_OFFSET}" == 0 || "${STREAM_OFFSET}" == 4 ]] || {
  echo "H1A2_STREAM_OFFSET must be 0 or 4" >&2
  exit 2
}

for path in "${PLANS}" "${MODEL}" "${CHECKPOINT}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]] || missing "${CHECKPOINT}/adapter_model.safetensors"
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
plan_cohort	${PLANS}
plans	256
streams	4
stream_offset	${STREAM_OFFSET}
checkpoint	${CHECKPOINT}
checkpoint_total_epoch	2
generation_schedule	exact-plan
temperature	0.7
refiner	${REFINER}
refiner_steps	800
stability_condition	none
rl	false
rerank	false
pair_split	192_train_64_validation_frozen_before_labels
EOF

sample_cell() {
  local devices="$1" port="$2" seed="$3" output="$4"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${CHECKPOINT}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 256 \
    --batch-size 1 \
    --preserve-prompt-sample-idx \
    --seed "${seed}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

refine_cell() {
  local devices="$1" port="$2" seed="$3" proposals="$4" output="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
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

run_stream() {
  local devices="$1" sample_port="$2" refine_port="$3" stream="$4"
  local dlm_seed=$((31017 + 1000 * stream))
  local refiner_seed=$((41017 + 1000 * stream))
  local output="${RUN_DIR}/stream${stream}"
  sample_cell "${devices}" "${sample_port}" "${dlm_seed}" "${output}/body"
  refine_cell "${devices}" "${refine_port}" "${refiner_seed}" \
    "${output}/body/proposal_graphs.pt" "${output}/refine"
  touch "${output}/_SUCCESS"
}

run_stream 0,1 31101 31111 $((STREAM_OFFSET + 0)) & p0=$!
run_stream 2,3 31102 31112 $((STREAM_OFFSET + 1)) & p1=$!
run_stream 4,5 31103 31113 $((STREAM_OFFSET + 2)) & p2=$!
run_stream 6,7 31104 31114 $((STREAM_OFFSET + 3)) & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for stream in $(seq "${STREAM_OFFSET}" $((STREAM_OFFSET + 3))); do
  [[ -f "${RUN_DIR}/stream${stream}/_SUCCESS" ]] || missing "${RUN_DIR}/stream${stream}/_SUCCESS"
done
touch "${RUN_DIR}/_SUCCESS"
log "Energy-pair multi-stream bodies completed: ${RUN_DIR}"

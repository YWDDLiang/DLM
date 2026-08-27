#!/usr/bin/env bash
#SBATCH --job-name=h1a2-dlm-raw1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=700G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-dlm-raw1000-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly TRAIN_RUN="${H1A2_SUFFICIENT_TRAIN_RUN:?export H1A2_SUFFICIENT_TRAIN_RUN}"
readonly COHORT_DIR="${H1A2_RAW1000_TWO500_DIR:?export H1A2_RAW1000_TWO500_DIR}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/dlm_sufficient_raw1000_${SLURM_JOB_ID}"
readonly DLM_SEED=17117
readonly REFINER_SEED=27117

readonly ROUND1="${COHORT_DIR}/round1_first500.jsonl"
readonly ROUND2="${COHORT_DIR}/round2_last500.jsonl"
readonly CKPT2="${TRAIN_RUN}/train/checkpoints/step-696"
readonly CKPT3="${TRAIN_RUN}/train/checkpoints/step-2392"

for path in "${MODEL}" "${REFINER}" "${CRYS}" "${ROUND1}" "${ROUND2}" "${CKPT2}" "${CKPT3}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for checkpoint in "${CKPT2}" "${CKPT3}"; do
  [[ -f "${checkpoint}/adapter_model.safetensors" ]] || missing "${checkpoint}/adapter_model.safetensors"
done
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
cohort_rule	first_1000_valid_source_order_split_first500_last500
round1	${ROUND1}
round2	${ROUND2}
round1_sha256	0167ee29383022900483ba3998021bb7344515d85bf73c7a4d6e9aeaa115a783
round2_sha256	368c2d86f035c7246dcc3f2bb6eeeba717e437706b1f6aa687c724f207f976c5
checkpoint_total_epoch2	${CKPT2}
checkpoint_total_epoch3	${CKPT3}
attempts_per_cell	500
global_attempts_per_checkpoint	1000
dlm_seed	${DLM_SEED}
refiner_seed	${REFINER_SEED}
temperature	0.7
schedule	exact-plan
stability_condition	none
rerank	false
EOF

sample_cell() {
  local devices="$1" port="$2" checkpoint="$3" plans="$4" output="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${plans}" \
    --num-samples 500 \
    --batch-size 1 \
    --preserve-prompt-sample-idx \
    --seed "${DLM_SEED}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --generation-schedule exact-plan
}

refine_cell() {
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
    --max-proposals 500
}

run_cell() {
  local devices="$1" sample_port="$2" refine_port="$3" checkpoint="$4" plans="$5" output="$6"
  sample_cell "${devices}" "${sample_port}" "${checkpoint}" "${plans}" "${output}/body"
  refine_cell "${devices}" "${refine_port}" "${output}/body/proposal_graphs.pt" "${output}/refine"
  touch "${output}/_SUCCESS"
}

run_cell 0,1 30901 30911 "${CKPT2}" "${ROUND1}" "${RUN_DIR}/epoch2/round1" &
pid0=$!
run_cell 2,3 30902 30912 "${CKPT2}" "${ROUND2}" "${RUN_DIR}/epoch2/round2" &
pid1=$!
run_cell 4,5 30903 30913 "${CKPT3}" "${ROUND1}" "${RUN_DIR}/epoch3/round1" &
pid2=$!
run_cell 6,7 30904 30914 "${CKPT3}" "${ROUND2}" "${RUN_DIR}/epoch3/round2" &
pid3=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

for epoch in epoch2 epoch3; do
  for round in round1 round2; do
    [[ -f "${RUN_DIR}/${epoch}/${round}/_SUCCESS" ]] || missing "${RUN_DIR}/${epoch}/${round}/_SUCCESS"
  done
done
touch "${RUN_DIR}/_SUCCESS"
log "DLM sufficient-base raw1000 two-round generation completed: ${RUN_DIR}"

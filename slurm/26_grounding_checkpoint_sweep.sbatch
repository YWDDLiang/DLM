#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-ckpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-ground-ckpt-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly B0="${H1A2_DLM_B0_PATH:?export H1A2_DLM_B0_PATH}"
readonly DATA="${H1A2_DLM_DATA_PATH:?export H1A2_DLM_DATA_PATH}"
readonly GROUNDED_DATA="${H1A2_GROUNDED_DATA_PATH:?export H1A2_GROUNDED_DATA_PATH}"
readonly PLANS="${H1A2_SCREEN_PLANS:?export H1A2_SCREEN_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/grounding_checkpoint_sweep_${SLURM_JOB_ID}"
readonly TRAIN_SEED=17017
readonly DLM_SEED=17117
readonly REFINER_SEED=27117
readonly CHECKPOINT_STEPS=(500 1000)

for path in "${MODEL}" "${B0}" "${DATA}" "${GROUNDED_DATA}" "${CRYS}" "${PLANS}" "${REFINER}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${RUN_DIR}"

train_arm() {
  local devices="$1" port="$2" data="$3" output="$4" grounding_weight="$5"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" --nproc_per_node=2 \
    "${ROOT}/src/scripts/llada_sft.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${B0}" \
    --data-dir "${data}" \
    --output-dir "${output}" \
    --representation dynamic_v1 \
    --max-length 382 \
    --answer-token-count 87 \
    --epochs 1 \
    --max-train-steps 1000 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 5e-5 \
    --save-steps 500 \
    --eval-steps 500 \
    --seed "${TRAIN_SEED}" \
    --counterfactual-grounding-weight "${grounding_weight}"
}

train_arm 0,1 30101 "${DATA}" "${RUN_DIR}/control" 0.0 &
control_pid=$!
train_arm 2,3 30102 "${GROUNDED_DATA}" "${RUN_DIR}/candidate" 0.1 &
candidate_pid=$!
wait "${control_pid}"
wait "${candidate_pid}"

for step in "${CHECKPOINT_STEPS[@]}"; do
  for arm in control candidate; do
    [[ -d "${RUN_DIR}/${arm}/checkpoints/step-${step}" ]] || missing "${RUN_DIR}/${arm}/checkpoints/step-${step}"
  done
done

printf 'training_seed\tdlm_seed\trefiner_seed\n%d\t%d\t%d\n' \
  "${TRAIN_SEED}" "${DLM_SEED}" "${REFINER_SEED}" >"${RUN_DIR}/scientific_seeds.tsv"

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

sample_wave() {
  local step="$1" port_base="$2"
  local label
  label="$(printf '%04d' "${step}")"
  sample_arm 0,1 "${port_base}" "${RUN_DIR}/control/checkpoints/step-${step}" \
    "${RUN_DIR}/step${label}/control/body" &
  control_pid=$!
  sample_arm 2,3 "$((port_base + 1))" "${RUN_DIR}/candidate/checkpoints/step-${step}" \
    "${RUN_DIR}/step${label}/candidate/body" &
  candidate_pid=$!
  wait "${control_pid}"
  wait "${candidate_pid}"
}

sample_wave 500 30201
sample_wave 1000 30203

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

refine_wave() {
  local step="$1" port_base="$2"
  local label
  label="$(printf '%04d' "${step}")"
  refine_arm 0,1 "${port_base}" "${RUN_DIR}/step${label}/control/body/proposal_graphs.pt" \
    "${RUN_DIR}/step${label}/control/refine" &
  control_pid=$!
  refine_arm 2,3 "$((port_base + 1))" "${RUN_DIR}/step${label}/candidate/body/proposal_graphs.pt" \
    "${RUN_DIR}/step${label}/candidate/refine" &
  candidate_pid=$!
  wait "${control_pid}"
  wait "${candidate_pid}"
}

refine_wave 500 30301
refine_wave 1000 30303

touch "${RUN_DIR}/_SUCCESS"
log "Grounding 500/1000 checkpoint sweep complete: ${RUN_DIR}"

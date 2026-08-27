#!/usr/bin/env bash
#SBATCH --job-name=h1a2-dlm-sufficient
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-dlm-sufficient-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly DATA="${H1A2_DLM_DATA_PATH:?export H1A2_DLM_DATA_PATH}"
readonly RESUME="${H1A2_RESUME_CHECKPOINT:?export H1A2_RESUME_CHECKPOINT}"
readonly RUN_DIR="${ROOT}/runs/dlm_sufficient_base_${SLURM_JOB_ID}"

for path in "${MODEL}" "${DATA}" "${RESUME}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${RESUME}/adapter_model.safetensors" ]] || missing "${RESUME}/adapter_model.safetensors"
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
initialization	${RESUME}
absolute_epoch_at_initialization	1.589623
one_epoch_optimizer_steps	1696
additional_optimizer_steps	2392
absolute_epoch_at_final	3.000000
world_size	4
batch_size_per_rank	1
gradient_accumulation	4
effective_batch	16
learning_rate	1e-5
lr_scheduler	cosine
stability_condition	none
preference_loss	off
planner	frozen
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30821 --nproc_per_node=4 \
  "${ROOT}/src/scripts/llada_sft.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${RESUME}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN_DIR}/train" \
  --representation dynamic_v1 \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 2 \
  --max-train-steps 2392 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 1e-5 \
  --lr-scheduler cosine \
  --warmup-steps 100 \
  --min-lr-ratio 0.1 \
  --save-steps 696 \
  --eval-steps 500 \
  --eval-max-batches 50 \
  --seed 17017 \
  --counterfactual-grounding-weight 0.0

for step in 696 2392; do
  [[ -f "${RUN_DIR}/train/checkpoints/step-${step}/adapter_model.safetensors" ]] || \
    missing "${RUN_DIR}/train/checkpoints/step-${step}/adapter_model.safetensors"
done
touch "${RUN_DIR}/_SUCCESS"
log "DLM sufficient-base continuation completed: ${RUN_DIR}"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-features
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=h1a2-ctv-features-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_CTV_CHECKPOINT:?export H1A2_CTV_CHECKPOINT}"
readonly TRAIN_ROLLOUT="${H1A2_CTV_TRAIN_ROLLOUT:?export H1A2_CTV_TRAIN_ROLLOUT}"
readonly VALIDATION_ROLLOUT="${H1A2_CTV_VALIDATION_ROLLOUT:?export H1A2_CTV_VALIDATION_ROLLOUT}"
readonly DATASET_AUDIT="${H1A2_CTV_DATASET_AUDIT:?export H1A2_CTV_DATASET_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/ctv_state_features_${SLURM_JOB_ID}"

for path in \
  "${MODEL}" \
  "${CHECKPOINT}/adapter_model.safetensors" \
  "${TRAIN_ROLLOUT}/_SUCCESS" \
  "${VALIDATION_ROLLOUT}/_SUCCESS" \
  "${DATASET_AUDIT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/src/scripts/extract_ctv_state_features.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${CHECKPOINT}" \
  --train-rollout-run "${TRAIN_ROLLOUT}" \
  --validation-rollout-run "${VALIDATION_ROLLOUT}" \
  --dataset-audit "${DATASET_AUDIT}" \
  --output-dir "${RUN_DIR}" \
  --projection-dim 256 \
  --projection-seed 73017 \
  --probability-atol 1e-5

[[ -f "${RUN_DIR}/_SUCCESS" ]] || missing "${RUN_DIR}/_SUCCESS"
log "CTV frozen state/action feature extraction finished: ${RUN_DIR}"

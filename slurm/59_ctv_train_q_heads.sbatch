#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-qheads
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=h1a2-ctv-qheads-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly FEATURES="${H1A2_CTV_FEATURES:?export H1A2_CTV_FEATURES}"
readonly TRAIN_LABELS="${H1A2_CTV_TRAIN_LABELS:?export H1A2_CTV_TRAIN_LABELS}"
readonly VALIDATION_LABELS="${H1A2_CTV_VALIDATION_LABELS:?export H1A2_CTV_VALIDATION_LABELS}"
readonly DATASET_AUDIT="${H1A2_CTV_DATASET_AUDIT:?export H1A2_CTV_DATASET_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/ctv_q_heads_${SLURM_JOB_ID}"
for path in "${FEATURES}" "${TRAIN_LABELS}" "${VALIDATION_LABELS}" "${DATASET_AUDIT}"; do
  [[ -f "${path}" ]] || missing "${path}"
done

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/src/scripts/train_ctv_q_heads.py" \
  --features "${FEATURES}" \
  --train-labels "${TRAIN_LABELS}" \
  --output-dir "${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/src/scripts/finalize_ctv_q_gate.py" \
  --features "${FEATURES}" \
  --q-heads "${RUN_DIR}/CTV_Q_HEADS.pt" \
  --validation-labels "${VALIDATION_LABELS}" \
  --dataset-audit "${DATASET_AUDIT}" \
  --output-dir "${RUN_DIR}/gate"

[[ -f "${RUN_DIR}/gate/_SUCCESS" ]] || missing "${RUN_DIR}/gate/_SUCCESS"
log "CTV two-head training finished: ${RUN_DIR}"

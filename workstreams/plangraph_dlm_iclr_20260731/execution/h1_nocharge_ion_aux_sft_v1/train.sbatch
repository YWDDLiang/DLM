#!/bin/bash
#SBATCH --job-name=h1-nc-train-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%A_%a_train.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%A_%a_train.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260806_h1_nocharge_ion_aux_sft_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${LEGACY_PYTHON:?missing training runtime}"
: "${SLURM_ARRAY_TASK_ID:?missing arm identity}"
case "${SLURM_ARRAY_TASK_ID}" in 0) ARM=c0 ;; 1) ARM=c1 ;; *) exit 3 ;; esac
test "${SLURM_JOB_PARTITION:-}" = gpu
test -f "${RUN_ROOT}/status/smoke_${ARM}_SUCCESS"
test ! -e "${RUN_ROOT}/training/${ARM}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/training" "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/train_${ARM}_exit_code.txt"
  touch "${RUN_ROOT}/status/train_${ARM}_FAILED"
  exit "${rc}"
}
trap on_error ERR

"${LEGACY_PYTHON}" scripts/llama_nocharge_ion_aux_sft.py \
  --model-path "${MODEL_PATH}" \
  --p0-adapter-path "${P0_ADAPTER}" \
  --data-dir "${RUN_ROOT}/data/sft/${ARM}" \
  --output-dir "${RUN_ROOT}/training/${ARM}" \
  --arm "${ARM}" \
  --max-length 512 --batch-size 1 --grad-accum 8 --max-updates 400 \
  --lr 0.000002 --weight-decay 0 --warmup-steps 25 --kl-beta 0.05 \
  --seed 26080617 --logging-steps 10 --eval-steps 100 --eval-max-batches 160 \
  --checkpoint-steps 100,200,300,400
test -f "${RUN_ROOT}/training/${ARM}/_SUCCESS"
test -f "${RUN_ROOT}/training/${ARM}/checkpoint-0400/adapter_location.json"
printf '0\n' > "${RUN_ROOT}/status/train_${ARM}_exit_code.txt"
touch "${RUN_ROOT}/status/train_${ARM}_SUCCESS"
echo "H1_NOCHARGE_TRAIN_PASS arm=${ARM} fixed_endpoint=checkpoint-0400"

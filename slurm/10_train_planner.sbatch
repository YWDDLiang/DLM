#!/usr/bin/env bash
#SBATCH --job-name=h1a2-train-planner
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-train-planner-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly MODEL="${REPO_ROOT}/checkpoints/planner/base"
readonly EPOCH1="${REPO_ROOT}/checkpoints/planner/epoch1"
readonly DATA="${REPO_ROOT}/data/planner"
readonly OUTPUT="${REPO_ROOT}/runs/train_planner_${SLURM_JOB_ID}"

[[ -d "${MODEL}" ]] || missing "${MODEL}"
[[ -d "${EPOCH1}" ]] || missing "${EPOCH1}"
[[ -f "${DATA}/train.jsonl" ]] || missing "${DATA}/train.jsonl"

python "${REPO_ROOT}/src/scripts/llama_formula_sft.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${EPOCH1}" \
  --data-dir "${DATA}" \
  --output-dir "${OUTPUT}" \
  --max-length 768 --epochs 1.0 --batch-size 1 --grad-accum 8 \
  --lr 2e-5 --weight-decay 0.0 --warmup-steps 100 \
  --logging-steps 20 --eval-steps 500 --eval-max-batches 50 \
  --seed 17 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --gradient-checkpointing


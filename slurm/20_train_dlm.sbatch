#!/usr/bin/env bash
#SBATCH --job-name=h1a2-train-dlm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=30:00:00
#SBATCH --output=h1a2-train-dlm-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

# The historical B0 global RNG seed is unrecorded. This command intentionally
# stops until the A800 audit recovers it or a documented release seed is set.
readonly GLOBAL_SEED="$(require_release_seed dlm_train_global)"
log "Using documented DLM release seed ${GLOBAL_SEED}"

readonly MODEL="${REPO_ROOT}/checkpoints/dlm/base"
readonly DATA="${REPO_ROOT}/data/dlm_r5_exact_length"
readonly OUTPUT="${REPO_ROOT}/runs/train_dlm_${SLURM_JOB_ID}"
[[ -d "${MODEL}" ]] || missing "${MODEL}"
[[ -f "${DATA}/train.jsonl" ]] || missing "${DATA}/train.jsonl"

log "The trainer source is installed. The seed CLI patch will be finalized after the A800 audit."
exit 2


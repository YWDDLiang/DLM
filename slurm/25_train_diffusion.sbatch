#!/usr/bin/env bash
#SBATCH --job-name=h1a2-train-diffusion
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=h1a2-train-diffusion-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly CONFIRMED_SEED="$(require_release_seed diffusion_train_confirmed)"
log "Using confirmed diffusion seed ${CONFIRMED_SEED}"
log "The model_494 training recipe is pending its A800 source/config audit."
exit 2


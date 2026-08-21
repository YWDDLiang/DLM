#!/usr/bin/env bash
#SBATCH --job-name=h1a2-body
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-body-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly SEED="$(require_release_seed dlm_inference)"
log "Using documented body inference seed ${SEED}"
log "Body inference seed wiring is pending the A800 seed audit."
exit 2


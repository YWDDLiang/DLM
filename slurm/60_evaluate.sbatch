#!/usr/bin/env bash
#SBATCH --job-name=h1a2-eval
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-eval-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

if [[ -z "${MP_API_KEY:-}" ]]; then
  log "MP_API_KEY is absent. Upstream outputs are retained; resume this stage after exporting the key."
  exit 3
fi

log "Evaluation source is installed; final relative input wiring awaits A800 assets."
exit 2


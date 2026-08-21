#!/usr/bin/env bash
#SBATCH --job-name=h1a2-env
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=h1a2-env-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ ! -f "${REPO_ROOT}/environment/environment.yml" ]]; then
  log "Environment placeholder is still active. Confirm packages on A800 first."
  exit 2
fi

activate_environment
python --version
python -m h1a2_repro.assets


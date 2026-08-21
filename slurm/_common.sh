#!/usr/bin/env bash

readonly SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SLURM_DIR}/.." && pwd)"
readonly RUNS_ROOT="${REPO_ROOT}/runs"
readonly CHECKPOINT_ACTION="${CHECKPOINT_ACTION:-train}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

missing() {
  log "Missing required asset: $1"
  log "See docs/PLACEHOLDER_ASSETS.md"
  return 2
}

activate_environment() {
  local env_name="${H1A2_CONDA_ENV:-h1a2-repro}"
  if ! command -v conda >/dev/null 2>&1; then
    log "Conda is not available; environment export is pending A800 confirmation"
    return 2
  fi
  local conda_base
  conda_base="$(conda info --base)"
  set +u
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
  set -u
}

require_release_seed() {
  local field="$1"
  python - "${field}" <<'PY'
import sys
from h1a2_repro.science import SEEDS

name = sys.argv[1]
value = getattr(SEEDS, name)
if value is None:
    raise SystemExit(
        f"Seed '{name}' is not recovered yet; see docs/SEEDS.md before running this stage"
    )
print(value)
PY
}


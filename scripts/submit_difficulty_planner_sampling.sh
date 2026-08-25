#!/usr/bin/env bash
set -Eeuo pipefail
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is required" >&2; exit 2; }
sbatch "${REPO_ROOT}/slurm/12_sample_difficulty_plans.sbatch"


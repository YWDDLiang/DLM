#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required" >&2
  exit 2
fi
sbatch "${REPO_ROOT}/slurm/21_grounding_candidate.sbatch"

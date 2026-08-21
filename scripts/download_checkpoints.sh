#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

download_one() {
  local label="$1"
  local url="$2"
  local destination="$3"
  if [[ -z "${url}" ]]; then
    echo "${label}: download URL is not published; see docs/PLACEHOLDER_ASSETS.md" >&2
    return 2
  fi
  mkdir -p "$(dirname -- "${destination}")"
  curl --fail --location "${url}" --output "${destination}"
}

status=0
download_one "Planner" "${H1A2_PLANNER_URL:-}" "${REPO_ROOT}/checkpoints/planner/checkpoint.tar.gz" || status=$?
download_one "DLM" "${H1A2_DLM_URL:-}" "${REPO_ROOT}/checkpoints/dlm/checkpoint.tar.gz" || status=$?
download_one "Diffusion" "${H1A2_DIFFUSION_URL:-}" "${REPO_ROOT}/checkpoints/diffusion/model_494.pt" || status=$?
exit "${status}"


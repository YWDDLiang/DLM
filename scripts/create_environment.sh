#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly ENV_FILE="${REPO_ROOT}/environment/environment.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "The A800 environment export has not been installed yet." >&2
  echo "See environment/README.md and docs/PLACEHOLDER_ASSETS.md." >&2
  exit 2
fi

conda env create -f "${ENV_FILE}"


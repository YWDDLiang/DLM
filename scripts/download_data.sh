#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DESTINATION="${REPO_ROOT}/data/mp20"
readonly URL="${H1A2_DATA_URL:-}"

if [[ -z "${URL}" ]]; then
  echo "The public MP-20 download URL is not published yet." >&2
  echo "Install the frozen split under data/mp20/ or set H1A2_DATA_URL." >&2
  exit 2
fi

mkdir -p "${DESTINATION}"
curl --fail --location "${URL}" --output "${DESTINATION}/mp20-release.tar.gz"
tar -xzf "${DESTINATION}/mp20-release.tar.gz" -C "${DESTINATION}"
echo "Downloaded data to ${DESTINATION}"


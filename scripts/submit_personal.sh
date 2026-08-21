#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONFIG="${1:-${REPO_ROOT}/configs/personal.example.json}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Personal config is missing: ${CONFIG}" >&2
  exit 2
fi

eval "$(python - "${CONFIG}" <<'PY'
import json
import shlex
import sys

cfg = json.load(open(sys.argv[1], encoding='utf-8'))
science = cfg['science']
runtime = cfg['runtime']
values = {
    'SAFE_AXIS': str(bool(science.get('safe_axis'))).lower(),
    'RESAMPLE_PLANS': str(bool(science.get('resample_plans'))).lower(),
    'CHECKPOINT_ACTION': str(runtime.get('checkpoint_action', 'train')),
    'H1A2_CONDA_ENV': str(runtime.get('conda_env', 'h1a2-personal')),
}
for key, value in values.items():
    print(f'export {key}={shlex.quote(value)}')
PY
)"

if [[ "${PERSONAL_ROUTE:-h1a2}" == "quick" ]]; then
  exec "${SCRIPT_DIR}/submit_quick_256x4.sh"
fi
exec "${SCRIPT_DIR}/submit_h1a2.sh"


#!/bin/bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
SITE="${SCRIPT_DIR}/site"
BASE_PYTHON_FILE="${SCRIPT_DIR}/base_python_path.txt"
test -d "${SITE}"
test -f "${BASE_PYTHON_FILE}"
IFS= read -r BASE_PYTHON < "${BASE_PYTHON_FILE}"
case "${BASE_PYTHON}" in
  /*|..|../*|*/..|*/../*) exit 3 ;;
esac
BASE_PYTHON="${SCRIPT_DIR}/${BASE_PYTHON}"
test -x "${BASE_PYTHON}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SITE}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BASE_PYTHON}" "$@"

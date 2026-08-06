#!/usr/bin/env bash
set -Eeuo pipefail

A100_DIR="${A100_DIR:-/home/ywliang/codex_work/diffsion_language_model_meets_diffusion}"
A800_USER="${A800_USER:-jiaosz}"
A800_HOST="${A800_HOST:-jump.gleamoe.com}"
A800_PORT="${A800_PORT:-7001}"
A800_DIR="${A800_DIR:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DRY_RUN="${DRY_RUN:-1}"
EXCLUDES="$(dirname "$0")/common_excludes.txt"

cd "${A100_DIR}"
flags=(-az --delete --exclude-from "${EXCLUDES}")
if [ "${DRY_RUN}" = "1" ]; then
  flags+=(-n)
fi

rsync "${flags[@]}" -e "ssh -p ${A800_PORT}" "${A100_DIR}/" "${A800_USER}@${A800_HOST}:${A800_DIR}/"

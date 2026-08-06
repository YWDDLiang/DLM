#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_DIR="${LOCAL_DIR:-/mnt/d/codex_work/ai4s/diffsion_language_model_meets_diffusion}"
A100_USER="${A100_USER:-ywliang}"
A100_HOST="${A100_HOST:-server.starteam.wang}"
A100_PORT="${A100_PORT:-2211}"
A100_DIR="${A100_DIR:-/home/ywliang/codex_work/diffsion_language_model_meets_diffusion}"
DRY_RUN="${DRY_RUN:-1}"
EXCLUDES="$(dirname "$0")/common_excludes.txt"

flags=(-az --delete --exclude-from "${EXCLUDES}")
if [ "${DRY_RUN}" = "1" ]; then
  flags+=(-n)
fi

ssh_cmd=(ssh -p "${A100_PORT}")
if [ -n "${A100_SSH_KEY:-}" ]; then
  ssh_cmd+=(-i "${A100_SSH_KEY}")
fi

rsync "${flags[@]}" -e "${ssh_cmd[*]}" "${LOCAL_DIR}/" "${A100_USER}@${A100_HOST}:${A100_DIR}/"

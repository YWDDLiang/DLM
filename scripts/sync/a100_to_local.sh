#!/usr/bin/env bash
set -Eeuo pipefail

if [ -z "${RUN_ID:-}" ]; then
  echo "RUN_ID is required" >&2
  exit 2
fi

LOCAL_DIR="${LOCAL_DIR:-/mnt/d/codex_work/ai4s/diffsion_language_model_meets_diffusion}"
A100_USER="${A100_USER:-ywliang}"
A100_HOST="${A100_HOST:-server.starteam.wang}"
A100_PORT="${A100_PORT:-2211}"
A100_DIR="${A100_DIR:-/home/ywliang/codex_work/diffsion_language_model_meets_diffusion}"

mkdir -p "${LOCAL_DIR}/runs/${RUN_ID}"
ssh_cmd=(ssh -p "${A100_PORT}")
if [ -n "${A100_SSH_KEY:-}" ]; then
  ssh_cmd+=(-i "${A100_SSH_KEY}")
fi

rsync -az \
  --include '*/' \
  --include 'logs/***' \
  --include 'notes/***' \
  --include '*.json' \
  --include '*.jsonl' \
  --include '*.md' \
  --exclude '*' \
  -e "${ssh_cmd[*]}" \
  "${A100_USER}@${A100_HOST}:${A100_DIR}/runs/${RUN_ID}/" \
  "${LOCAL_DIR}/runs/${RUN_ID}/"

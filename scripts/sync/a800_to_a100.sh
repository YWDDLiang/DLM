#!/usr/bin/env bash
set -Eeuo pipefail

if [ -z "${RUN_ID:-}" ]; then
  echo "RUN_ID is required" >&2
  exit 2
fi

A100_DIR="${A100_DIR:-/home/ywliang/codex_work/diffsion_language_model_meets_diffusion}"
A800_USER="${A800_USER:-jiaosz}"
A800_HOST="${A800_HOST:-jump.gleamoe.com}"
A800_PORT="${A800_PORT:-7001}"
A800_DIR="${A800_DIR:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"

mkdir -p "${A100_DIR}/runs/${RUN_ID}"
rsync -az \
  --include '*/' \
  --include 'logs/***' \
  --include 'notes/***' \
  --include '*.json' \
  --include '*.jsonl' \
  --include '*.md' \
  --exclude '*' \
  -e "ssh -p ${A800_PORT}" \
  "${A800_USER}@${A800_HOST}:${A800_DIR}/runs/${RUN_ID}/" \
  "${A100_DIR}/runs/${RUN_ID}/"

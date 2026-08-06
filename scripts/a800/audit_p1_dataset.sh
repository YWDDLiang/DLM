#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 OUTPUT_JSON" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
P1_ROOT="${WQ_P1_ROOT:-${PROJECT_ROOT}/data/wqcodiff/p1_v3}"
OUTPUT="$1"
cd "${PROJECT_ROOT}"

if [ -e "${OUTPUT}" ]; then
  echo "Immutable P1 audit output already exists: ${OUTPUT}" >&2
  exit 3
fi
arguments=()
for split in train val test; do
  for index in $(seq 0 7); do
    path="${P1_ROOT}/${split}.part-$(printf '%03d' "${index}")-of-008.jsonl"
    if [ ! -f "${path}" ]; then
      echo "Missing P1 shard: ${path}" >&2
      exit 4
    fi
    arguments+=(--split "${split}=${path}")
  done
done

python -m crystal_dlm.wqcodiff \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v3.yaml \
  dataset-audit \
  "${arguments[@]}" \
  --expected-total 45229 \
  --output "${OUTPUT}"

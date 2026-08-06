#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260526_mp20_ehull0_composition_all_splits}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes" "runs/${RUN_ID}/logs"

for split in train val test; do
  python scripts/analyze_mp20_train_distribution.py \
    --csv "reference/crysllmgen/data/mp_20/${split}.csv" \
    --split "${split}" \
    --output-json "runs/${RUN_ID}/notes/mp20_${split}_distribution.json" \
    --output-md "runs/${RUN_ID}/notes/mp20_${split}_distribution.md" \
    > "runs/${RUN_ID}/logs/${split}_distribution_stdout.json"
done

python scripts/analyze_mp20_ehull0_composition.py \
  --run-dir "runs/${RUN_ID}" \
  --output-json "runs/${RUN_ID}/notes/mp20_ehull0_composition_summary.json" \
  --output-md "runs/${RUN_ID}/notes/mp20_ehull0_composition_summary.md"

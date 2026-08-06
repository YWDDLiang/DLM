#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"

TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-default}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"

RUN_DIR="runs/${RUN_ID}"
SAMPLE_DIR="${RUN_DIR}/outputs/sample256"
mkdir -p "${SAMPLE_DIR}" "${RUN_DIR}/notes"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "temperature": float("${TEMPERATURE}"),
    "sample_count": int("${SAMPLE_COUNT}"),
    "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "master_port": int("${MASTER_PORT}"),
}, indent=2) + "\n")
PY

sample_args=(
  --checkpoint-path "${CHECKPOINT_PATH}"
  --output-dir "${SAMPLE_DIR}"
  --num-samples "${SAMPLE_COUNT}"
  --batch-size "${SAMPLE_BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
)
if [ "${GENERATION_SCHEDULE}" != "default" ]; then
  sample_args+=(--generation-schedule "${GENERATION_SCHEDULE}")
fi

torchrun --master_port "${MASTER_PORT}" --nproc_per_node=2 scripts/sample_llada_crystals.py "${sample_args[@]}"

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_distribution.json" \
  --output-md "${RUN_DIR}/notes/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_composition.json" \
  --output-md "${RUN_DIR}/notes/sample256_composition.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_failure_modes.json" \
  --output-md "${RUN_DIR}/notes/sample256_failure_modes.md"

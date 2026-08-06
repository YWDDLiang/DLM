#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-${1:-}}"
ROLLOUT_CEPO_JSONL="${ROLLOUT_CEPO_JSONL:-${2:-}}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:?BEST_CHECKPOINT is required}"
ROLLOUT_CEPO_JSONL="${ROLLOUT_CEPO_JSONL:?ROLLOUT_CEPO_JSONL is required}"

TEMPERATURE="${TEMPERATURE:-0.7}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
LR="${LR:-2e-7}"
CLIP_EPS="${CLIP_EPS:-0.15}"
BETA="${BETA:-0.02}"
TRACE_SHRINK="${TRACE_SHRINK:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"

RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/outputs/rollout" "${RUN_DIR}/outputs/tracerl" "${RUN_DIR}/outputs/sample256" "${RUN_DIR}/notes"
cp "${ROLLOUT_CEPO_JSONL}" "${RUN_DIR}/outputs/rollout/rollout_cepo.jsonl"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "source_rollout_cepo_jsonl": "${ROLLOUT_CEPO_JSONL}",
    "temperature": float("${TEMPERATURE}"),
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "rl_mode": "reason_aware_cepo_lite_resume_trace",
    "lr": float("${LR}"),
    "clip_eps": float("${CLIP_EPS}"),
    "beta": float("${BETA}"),
    "trace_shrink": int("${TRACE_SHRINK}"),
    "max_train_steps": int("${MAX_TRAIN_STEPS}"),
}, indent=2) + "\\n")
PY

torchrun --nproc_per_node=2 scripts/llada_trace_rl.py \
  --checkpoint-path "${BEST_CHECKPOINT}" \
  --rollout-jsonl "${RUN_DIR}/outputs/rollout/rollout_cepo.jsonl" \
  --output-dir "${RUN_DIR}/outputs/tracerl" \
  --lr "${LR}" \
  --clip-eps "${CLIP_EPS}" \
  --beta "${BETA}" \
  --trace-shrink "${TRACE_SHRINK}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --logging-steps 5 \
  --save-steps "${MAX_TRAIN_STEPS}"

torchrun --nproc_per_node=2 scripts/sample_llada_crystals.py \
  --checkpoint-path "${RUN_DIR}/outputs/tracerl/final" \
  --output-dir "${RUN_DIR}/outputs/sample256" \
  --num-samples "${SAMPLE_COUNT}" \
  --batch-size "${SAMPLE_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --generation-schedule "${GENERATION_SCHEDULE}"

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --failure-jsonl "${RUN_DIR}/outputs/sample256/failure_cases.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_distribution.json" \
  --output-md "${RUN_DIR}/notes/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_composition.json" \
  --output-md "${RUN_DIR}/notes/sample256_composition.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_failure_modes.json" \
  --output-md "${RUN_DIR}/notes/sample256_failure_modes.md"

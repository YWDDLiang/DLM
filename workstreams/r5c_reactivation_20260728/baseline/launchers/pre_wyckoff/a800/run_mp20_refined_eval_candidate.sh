#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"

GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-default}"
REPRESENTATION="${REPRESENTATION:-fixed_slot}"
COMPRESSED_TOKEN_CONFIG="${COMPRESSED_TOKEN_CONFIG:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1200}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
BASE_MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
SAMPLE_MASTER_PORT="${SAMPLE_MASTER_PORT:-${BASE_MASTER_PORT}}"
REFINE_MASTER_PORT="${REFINE_MASTER_PORT:-$((BASE_MASTER_PORT + 1))}"

RUN_DIR="runs/${RUN_ID}"
SAMPLE_DIR="${RUN_DIR}/outputs/sample1000"
REFINED_DIR="${RUN_DIR}/outputs/refined1000"
NOTES_DIR="${RUN_DIR}/notes"
mkdir -p "${SAMPLE_DIR}" "${REFINED_DIR}" "${NOTES_DIR}"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "representation": "${REPRESENTATION}",
    "compressed_token_config": "${COMPRESSED_TOKEN_CONFIG}",
    "temperature": float("${TEMPERATURE}"),
    "target_graph_success": int("${TARGET_GRAPH_SUCCESS}"),
    "max_attempts": int("${MAX_ATTEMPTS}"),
    "diff_steps": int("${DIFF_STEPS}"),
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "buffer_policy": "raw outputs are diagnostics only; SFT buffer requires diffusion-refined accepted data",
}, indent=2) + "\n")
PY

sample_args=(
  --checkpoint-path "${CHECKPOINT_PATH}"
  --representation "${REPRESENTATION}"
  --output-dir "${SAMPLE_DIR}"
  --target-graph-success "${TARGET_GRAPH_SUCCESS}"
  --max-attempts "${MAX_ATTEMPTS}"
  --num-samples "${MAX_ATTEMPTS}"
  --batch-size "${SAMPLE_BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
)
if [ -n "${COMPRESSED_TOKEN_CONFIG}" ]; then
  sample_args+=(--compressed-token-config "${COMPRESSED_TOKEN_CONFIG}")
fi
if [ "${GENERATION_SCHEDULE}" != "default" ]; then
  sample_args+=(--generation-schedule "${GENERATION_SCHEDULE}")
fi

torchrun --nproc_per_node=2 --master_port "${SAMPLE_MASTER_PORT}" scripts/sample_llada_crystals.py "${sample_args[@]}"

sample_analysis_args=(
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl"
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl"
  --representation "${REPRESENTATION}"
  --output-json "${NOTES_DIR}/sample1000_distribution.json"
  --output-md "${NOTES_DIR}/sample1000_distribution.md"
)
if [ -n "${COMPRESSED_TOKEN_CONFIG}" ]; then
  sample_analysis_args+=(--compressed-token-config "${COMPRESSED_TOKEN_CONFIG}")
fi
python scripts/analyze_sample_outputs.py "${sample_analysis_args[@]}"

composition_raw_args=(
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl"
  --representation "${REPRESENTATION}"
  --refined-world-size "${REFINED_WORLD_SIZE}"
  --output-json "${NOTES_DIR}/sample1000_composition_raw.json"
  --output-md "${NOTES_DIR}/sample1000_composition_raw.md"
)
if [ -n "${COMPRESSED_TOKEN_CONFIG}" ]; then
  composition_raw_args+=(--compressed-token-config "${COMPRESSED_TOKEN_CONFIG}")
fi
python scripts/analyze_composition_validity.py "${composition_raw_args[@]}"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample1000_failure_modes_raw.json" \
  --output-md "${NOTES_DIR}/sample1000_failure_modes_raw.md"

torchrun --nproc_per_node=2 --master_port "${REFINE_MASTER_PORT}" scripts/refine_dlm_with_crysllmgen.py \
  --proposal-graphs "${SAMPLE_DIR}/proposal_graphs.pt" \
  --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
  --output-dir "${REFINED_DIR}" \
  --batch-size "${REFINE_BATCH_SIZE}" \
  --diff-steps "${DIFF_STEPS}" \
  --max-proposals "${TARGET_GRAPH_SUCCESS}"

python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED_DIR}" \
  --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"

composition_refined_args=(
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl"
  --representation "${REPRESENTATION}"
  --refined-pt "${REFINED_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt"
  --refined-world-size "${REFINED_WORLD_SIZE}"
  --output-json "${NOTES_DIR}/composition1000.json"
  --output-md "${NOTES_DIR}/composition1000.md"
)
if [ -n "${COMPRESSED_TOKEN_CONFIG}" ]; then
  composition_refined_args+=(--compressed-token-config "${COMPRESSED_TOKEN_CONFIG}")
fi
python scripts/analyze_composition_validity.py "${composition_refined_args[@]}"

python scripts/evaluate_mp20_candidate_gate.py \
  --mode refined1000 \
  --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/composition1000.json" \
  --composition-key refined_pt \
  --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
  --min-comp-valid 0.90 \
  --min-strict-valid 0.50 \
  --max-single-element 0.10 \
  --max-pbc-duplicate 0.0 \
  --output-json "${NOTES_DIR}/refined1000_gate.json"

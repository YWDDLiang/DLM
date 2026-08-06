#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
CANDIDATE_NAME="${CANDIDATE_NAME:?CANDIDATE_NAME is required}"

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"

cd "${PROJECT_ROOT}"

BRANCH_DIR="runs/${RUN_ID}/outputs/${MODEL_LABEL}"
EVAL_DIR="${BRANCH_DIR}/manual_checkpoint_eval/${CANDIDATE_NAME}"
SAMPLE_DIR="${EVAL_DIR}/sample${NUM_SAMPLES}"
NOTES_DIR="${EVAL_DIR}/notes"
LOG_DIR="runs/${RUN_ID}/logs"
mkdir -p "${SAMPLE_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

echo "===== CHECKPOINT SMOKE EVAL ====="
echo "date=$(date '+%F %T %Z')"
echo "model_label=${MODEL_LABEL}"
echo "model_path=${MODEL_PATH}"
echo "checkpoint_path=${CHECKPOINT_PATH}"
echo "candidate_name=${CANDIDATE_NAME}"
echo "num_samples=${NUM_SAMPLES}"
echo "temperature=${TEMPERATURE}"
echo "generation_schedule=${GENERATION_SCHEDULE}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" scripts/sample_llada_crystals.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${SAMPLE_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${SAMPLE_BATCH_SIZE}" \
  --block-length 1 \
  --temperature "${TEMPERATURE}" \
  --generation-schedule "${GENERATION_SCHEDULE}" \
  --schema-logit-mask \
  --prefill-slot-tokens \
  --atom-count-grammar-mask \
  --duplicate-coordinate-mask \
  --lattice-volume-mask

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/sample${NUM_SAMPLES}_distribution.json" \
  --output-md "${NOTES_DIR}/sample${NUM_SAMPLES}_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample${NUM_SAMPLES}_composition.json" \
  --output-md "${NOTES_DIR}/sample${NUM_SAMPLES}_composition.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample${NUM_SAMPLES}_failure_modes.json" \
  --output-md "${NOTES_DIR}/sample${NUM_SAMPLES}_failure_modes.md"

SUMMARY_JSON="${NOTES_DIR}/sample${NUM_SAMPLES}_smoke_summary.json"
export SAMPLE_DIR NOTES_DIR SUMMARY_JSON NUM_SAMPLES MODEL_LABEL CANDIDATE_NAME CHECKPOINT_PATH
python - <<'PY'
import json
import os
from pathlib import Path

sample_dir = Path(os.environ["SAMPLE_DIR"])
notes_dir = Path(os.environ["NOTES_DIR"])
num_samples = os.environ["NUM_SAMPLES"]

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

sample_metrics = read_json(sample_dir / "sample_metrics.json")
composition = read_json(notes_dir / f"sample{num_samples}_composition.json")
failure = read_json(notes_dir / f"sample{num_samples}_failure_modes.json")
raw_comp = composition.get("raw_jsonl", composition)

payload = {
    "model_label": os.environ["MODEL_LABEL"],
    "candidate_name": os.environ["CANDIDATE_NAME"],
    "checkpoint_path": os.environ["CHECKPOINT_PATH"],
    "sample_metrics": str(sample_dir / "sample_metrics.json"),
    "composition": str(notes_dir / f"sample{num_samples}_composition.json"),
    "failure_modes": str(notes_dir / f"sample{num_samples}_failure_modes.json"),
    "parse_rate": sample_metrics.get("parse_rate", failure.get("parse_rate")),
    "graph_acceptance_rate": sample_metrics.get("graph_acceptance_rate", sample_metrics.get("raw_graph_rate")),
    "comp_valid": raw_comp.get("comp_valid_rate", failure.get("comp_valid_rate")),
    "strict_valid": failure.get("strict_valid_rate"),
    "single_element": (
        failure.get("reason_counts", {}).get("single_element_shortcut", 0)
        / max(1, failure.get("total_rows", raw_comp.get("count", 0)))
    ),
    "all_metal": (
        failure.get("reason_counts", {}).get("all_metal_shortcut", 0)
        / max(1, failure.get("total_rows", raw_comp.get("count", 0)))
    ),
    "pbc_duplicate": failure.get("pbc_equivalent_duplicate_fraction", raw_comp.get("pbc_equivalent_duplicate_fraction")),
    "reason_counts": failure.get("reason_counts", raw_comp.get("reason_counts")),
    "headline": failure.get("headline"),
}
Path(os.environ["SUMMARY_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

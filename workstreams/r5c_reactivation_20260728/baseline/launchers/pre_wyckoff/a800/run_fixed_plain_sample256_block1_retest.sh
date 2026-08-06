#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260525_234500-fixedplain-1e-smoke}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MIN_PARSE_RATE="${MIN_PARSE_RATE:-0.98}"
MIN_GRAPH_ACCEPTANCE="${MIN_GRAPH_ACCEPTANCE:-0.95}"
MIN_COMP_VALID="${MIN_COMP_VALID:-0.88}"
MIN_STRICT_VALID="${MIN_STRICT_VALID:-0.30}"
MAX_SINGLE_ELEMENT="${MAX_SINGLE_ELEMENT:-0.10}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs/fixed_plain_block1/sample256"
CHECKPOINT_PATH="runs/${SOURCE_RUN_ID}/outputs/fixed_plain/sft_epoch1/final"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((BASE_MASTER_PORT + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "===== COMMAND $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  printf '%q ' "$@" | tee -a "${log_file}"
  echo | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== STATUS ${status} $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  return "${status}"
}

if [[ ! -s "${CHECKPOINT_PATH}/adapter_model.safetensors" ]]; then
  echo "Missing source checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

run_logged "${LOG_DIR}/preflight.log" \
  python -m py_compile \
    crystal_dlm/fixed_plain.py \
    scripts/sample_llada_fixed_plain.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_mp20_candidate_gate.py

run_logged "${LOG_DIR}/unittest.log" \
  python -m unittest tests.test_fixed_plain

cat > "${NOTES_DIR}/sample256_block1_config.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "source_run_id": "${SOURCE_RUN_ID}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "model_path": "${MODEL_PATH}",
  "representation": "fixed_plain",
  "temperature": ${TEMPERATURE},
  "num_samples": ${SMOKE_SAMPLES},
  "block_lengths": {
    "count": 1,
    "lattice": 1,
    "elements": 1,
    "coords": 1
  },
  "note": "Retest fixed_plain final checkpoint with old successful block_length=1 behavior and short-coordinate normalization."
}
JSON

next_port
run_logged "${LOG_DIR}/sample256_block1.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_fixed_plain.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output-dir "${OUT_DIR}" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --count-gen-length 8 \
    --count-block-length 1 \
    --lattice-gen-length 48 \
    --lattice-block-length 1 \
    --elements-tokens-per-site 6 \
    --elements-extra-tokens 8 \
    --elements-block-length 1 \
    --coords-tokens-per-site 12 \
    --coords-extra-tokens 8 \
    --coords-block-length 1 \
    --coords-max-gen-length 288

COMPOSITION_KEY="raw_jsonl"
if [[ -s "${OUT_DIR}/raw_dlm_samples.pt" ]]; then
  COMPOSITION_KEY="raw_pt"
  run_logged "${LOG_DIR}/sample256_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${OUT_DIR}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${OUT_DIR}/raw_generations.jsonl" \
      --representation fixed_plain \
      --output-json "${NOTES_DIR}/sample256_composition.json" \
      --output-md "${NOTES_DIR}/sample256_composition.md"
else
  run_logged "${LOG_DIR}/sample256_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-generations-jsonl "${OUT_DIR}/raw_generations.jsonl" \
      --representation fixed_plain \
      --output-json "${NOTES_DIR}/sample256_composition.json" \
      --output-md "${NOTES_DIR}/sample256_composition.md"
fi

run_logged "${LOG_DIR}/sample256_gate.log" \
  python scripts/evaluate_mp20_candidate_gate.py \
    --mode smoke256 \
    --sample-metrics "${OUT_DIR}/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/sample256_composition.json" \
    --composition-key "${COMPOSITION_KEY}" \
    --min-parse-rate "${MIN_PARSE_RATE}" \
    --min-graph-acceptance "${MIN_GRAPH_ACCEPTANCE}" \
    --min-comp-valid "${MIN_COMP_VALID}" \
    --min-strict-valid "${MIN_STRICT_VALID}" \
    --max-single-element "${MAX_SINGLE_ELEMENT}" \
    --max-pbc-duplicate 0.0 \
    --output-json "${NOTES_DIR}/sample256_gate.json"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
out = Path("${OUT_DIR}")
payload = {
    "config": json.loads((notes / "sample256_block1_config.json").read_text(encoding="utf-8")),
    "sample256": json.loads((out / "sample_metrics.json").read_text(encoding="utf-8")),
    "composition256": json.loads((notes / "sample256_composition.json").read_text(encoding="utf-8")),
    "gate256": json.loads((notes / "sample256_gate.json").read_text(encoding="utf-8")),
}
(notes / "sample256_block1_report.md").write_text(
    "# Fixed-Plain Block1 Retest\\n\\n~~~json\\n"
    + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    + "\\n~~~\\n",
    encoding="utf-8",
)
PY

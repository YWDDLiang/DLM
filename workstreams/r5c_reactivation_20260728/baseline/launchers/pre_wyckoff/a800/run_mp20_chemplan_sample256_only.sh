#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260527_chemplan_bridge_r3b_fixsample256}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/20260527_chemplan_bridge_r3b/outputs/stage_a/final}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PLAN_TEMPERATURE="${PLAN_TEMPERATURE:-0.7}"
PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-96}"
PLAN_BLOCK_LENGTH="${PLAN_BLOCK_LENGTH:-16}"
PLAN_STEPS="${PLAN_STEPS:-96}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT:-4}}"
MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.88}"
MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.40}"
MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
MAX_PBC_DUPLICATE_256="${MAX_PBC_DUPLICATE_256:-0.0}"
MASTER_PORT_BASE="${MASTER_PORT:-$((25000 + (${SLURM_JOB_ID:-0} % 22000)))}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
SAMPLE_DIR="${OUT_DIR}/sample256"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${SAMPLE_DIR}"

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

python - <<PY
import json
from pathlib import Path
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps({
  "run_id": "${RUN_ID}",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "model_path": "${MODEL_PATH}",
  "temperature": float("${TEMPERATURE}"),
  "plan_temperature": float("${PLAN_TEMPERATURE}"),
  "plan_gen_length": int("${PLAN_GEN_LENGTH}"),
  "plan_block_length": int("${PLAN_BLOCK_LENGTH}"),
  "plan_steps": int("${PLAN_STEPS}"),
  "sampler": "chemical_plan_two_stage_fixed_text_field",
  "plan_ban_crystal_special_tokens": True,
}, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT_BASE}" scripts/sample_llada_chemical_plan_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output-dir "${SAMPLE_DIR}" \
    --num-samples 256 \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --block-length 1 \
    --temperature "${TEMPERATURE}" \
    --plan-temperature "${PLAN_TEMPERATURE}" \
    --plan-gen-length "${PLAN_GEN_LENGTH}" \
    --plan-block-length "${PLAN_BLOCK_LENGTH}" \
    --plan-steps "${PLAN_STEPS}" \
    --generation-schedule "${GENERATION_SCHEDULE}" \
    --schema-logit-mask \
    --prefill-slot-tokens \
    --atom-count-grammar-mask \
    --duplicate-coordinate-mask \
    --lattice-volume-mask \
    --plan-ban-crystal-special-tokens

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/sample256_distribution.json" \
  --output-md "${NOTES_DIR}/sample256_distribution.md"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample256_composition.json" \
  --output-md "${NOTES_DIR}/sample256_composition.md"
python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample256_failure_modes.json" \
  --output-md "${NOTES_DIR}/sample256_failure_modes.md"
python scripts/evaluate_mp20_candidate_gate.py \
  --mode smoke256 \
  --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/sample256_composition.json" \
  --composition-key raw_jsonl \
  --min-parse-rate 0.98 \
  --min-graph-acceptance 0.95 \
  --min-comp-valid "${MIN_COMP_VALID_256}" \
  --min-strict-valid "${MIN_STRICT_VALID_256}" \
  --max-single-element "${MAX_SINGLE_ELEMENT_256}" \
  --max-pbc-duplicate "${MAX_PBC_DUPLICATE_256}" \
  --output-json "${NOTES_DIR}/sample256_gate.json"

python - <<PY
import json
from collections import Counter
from pathlib import Path
raw = Path("${SAMPLE_DIR}/raw_generations.jsonl")
plans = []
if raw.exists():
    for line in raw.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            plans.append(str(row.get("plan") or ""))
fields = Counter()
special_plan_count = 0
for plan in plans:
    if "<N_" in plan or "<E_" in plan or "<X_" in plan:
        special_plan_count += 1
    for label in ("formula:", "composition:", "composition_reason:", "chemistry:", "stability_hint:", "geometry_hint:", "crystal tokens:"):
        if label in plan:
            fields[label] += 1
payload = {
    "plan_count": len(plans),
    "special_token_plan_count": special_plan_count,
    "field_presence": dict(fields),
    "examples": plans[:8],
}
Path("${NOTES_DIR}/plan_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
PY

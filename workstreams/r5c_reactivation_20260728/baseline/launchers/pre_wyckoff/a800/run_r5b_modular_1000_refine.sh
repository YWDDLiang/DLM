#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/${SOURCE_RUN_ID}/outputs/r5b_modular_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS_1000="${MAX_ATTEMPTS_1000:-1800}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MODULE_STYLE="${MODULE_STYLE:-composition}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
SAMPLE_DIR="${OUT_DIR}/r5b_sample1000"
REFINED_DIR="${OUT_DIR}/r5b_refined1000"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${SAMPLE_DIR}" "${REFINED_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((21000 + (${SLURM_JOB_ID:-0} % 30000)))}"
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

python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "source_run_id": "${SOURCE_RUN_ID}",
    "model_path": "${MODEL_PATH}",
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "target_graph_success": int("${TARGET_GRAPH_SUCCESS}"),
    "max_attempts_1000": int("${MAX_ATTEMPTS_1000}"),
    "temperature": float("${TEMPERATURE}"),
    "module_style": "${MODULE_STYLE}",
    "gate": {
        "post_refine_comp_valid_min": 0.90,
        "crysllmgen_cov_recall_min": 90.0,
        "crysllmgen_struct_valid_min": 99.0,
    },
}
Path("${NOTES_DIR}/r5b_1000_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/sample_llada_crysllmgen_modular.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py

next_port
run_logged "${LOG_DIR}/r5b_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_crysllmgen_modular.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output-dir "${SAMPLE_DIR}" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS_1000}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --module-style "${MODULE_STYLE}"

run_logged "${LOG_DIR}/r5b_sample1000_composition_raw.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --representation crysllmgen_text \
    --output-json "${NOTES_DIR}/r5b_sample1000_composition_raw.json" \
    --output-md "${NOTES_DIR}/r5b_sample1000_composition_raw.md"

next_port
run_logged "${LOG_DIR}/r5b_refined1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${SAMPLE_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps 800 \
    --max-proposals "${TARGET_GRAPH_SUCCESS}"

REFINED_PT=$(python - <<PY
from pathlib import Path
candidates = sorted(Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt"), key=lambda p: (p.stat().st_size, p.name), reverse=True)
candidates = [p for p in candidates if ".rank" not in p.name]
print(candidates[0] if candidates else "")
PY
)

run_logged "${LOG_DIR}/r5b_crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"

run_logged "${LOG_DIR}/r5b_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --refined-pt "${REFINED_PT}" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --representation crysllmgen_text \
    --refined-world-size "${NPROC_PER_NODE}" \
    --output-json "${NOTES_DIR}/composition1000.json" \
    --output-md "${NOTES_DIR}/composition1000.md"

run_logged "${LOG_DIR}/r5b_refined1000_gate.log" \
  python scripts/evaluate_r5b_gate.py \
    --mode refined1000 \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/composition1000.json" \
    --composition-key refined_pt \
    --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
    --min-comp-valid 0.90 \
    --min-crys-comp-valid 90.0 \
    --min-crys-struct-valid 99.0 \
    --min-crys-cov-recall 90.0 \
    --output-json "${NOTES_DIR}/refined1000_gate.json"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
sample = json.loads(Path("${SAMPLE_DIR}/sample_metrics.json").read_text(encoding="utf-8"))
comp = json.loads((notes / "composition1000.json").read_text(encoding="utf-8"))
crys = json.loads((notes / "crysllmgen_metrics1000.json").read_text(encoding="utf-8"))
gate = json.loads((notes / "refined1000_gate.json").read_text(encoding="utf-8"))
summary = comp.get("refined_pt") or comp.get("raw_pt") or {}
lines = [
    "# R5-B Modular 1000 Refined Gate",
    "",
    f"- RUN_ID: ${RUN_ID}",
    f"- source RUN_ID: ${SOURCE_RUN_ID}",
    f"- target graph success: ${TARGET_GRAPH_SUCCESS}",
    f"- sample graph acceptance: {sample.get('graph_acceptance_rate')}",
    f"- refined comp_valid: {summary.get('comp_valid_rate')}",
    f"- CrysLLMGen metrics: {json.dumps(crys.get('metrics', {}), sort_keys=True)}",
    f"- gate passed: {gate.get('passed')}",
    "",
    "## Gate",
    "",
    "```json",
    json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True),
    "```",
]
(notes / "r5b_1000_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

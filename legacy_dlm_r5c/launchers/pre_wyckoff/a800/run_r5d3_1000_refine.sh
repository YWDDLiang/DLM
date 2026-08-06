#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PLAN_PARENT_RUN_ID="${PLAN_PARENT_RUN_ID:-20260530_012145-r5d3-plancompact-256}"
PLAN_CHECKPOINT_PATH="${PLAN_CHECKPOINT_PATH:-runs/${PLAN_PARENT_RUN_ID}/outputs/r5d3_plan_compact_sft/final}"
PLAN_STATS_JSON="${PLAN_STATS_JSON:-data/dlm_sft/mp_20_r5_plan_compact/stats.json}"
BODY_PARENT_RUN_ID="${BODY_PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
BODY_CHECKPOINT_PATH="${BODY_CHECKPOINT_PATH:-runs/${BODY_PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PLAN_BATCH_SIZE="${PLAN_BATCH_SIZE:-8}"
BODY_BATCH_SIZE="${BODY_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
PLAN_SAMPLES="${PLAN_SAMPLES:-1100}"
BODY_SAMPLES="${BODY_SAMPLES:-1100}"
TARGET_REFINED="${TARGET_REFINED:-1000}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PLAN_MAX_ATTEMPT_FACTOR="${PLAN_MAX_ATTEMPT_FACTOR:-12}"
PLAN_FORMULA_CAP="${PLAN_FORMULA_CAP:-2}"
PLAN_N_CAP_FRACTION="${PLAN_N_CAP_FRACTION:-0.34}"
DIFF_STEPS="${DIFF_STEPS:-800}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
PLAN_SAMPLE_DIR="${OUT_DIR}/r5d3_plan_sample1000"
BODY_SAMPLE_DIR="${OUT_DIR}/r5d3_body_sample1000"
REFINED_DIR="${OUT_DIR}/r5d3_refined1000"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${PLAN_SAMPLE_DIR}" "${BODY_SAMPLE_DIR}" "${REFINED_DIR}"

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

assert_gate_passed() {
  local gate_json="$1"
  local label="$2"
  python - "$gate_json" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("passed"):
    print(f"{label} gate failed: {payload.get('failures')}", file=sys.stderr)
    raise SystemExit(1)
print(f"{label} gate passed")
PY
}

python - <<PY
import json
from pathlib import Path

payload = {
    "run_id": "${RUN_ID}",
    "stage": "r5d3_plan_compact_exact_body_refined1000",
    "plan_parent_run_id": "${PLAN_PARENT_RUN_ID}",
    "plan_checkpoint_path": "${PLAN_CHECKPOINT_PATH}",
    "plan_stats_json": "${PLAN_STATS_JSON}",
    "body_parent_run_id": "${BODY_PARENT_RUN_ID}",
    "body_checkpoint_path": "${BODY_CHECKPOINT_PATH}",
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "model_path": "${MODEL_PATH}",
    "plan_samples": int("${PLAN_SAMPLES}"),
    "body_samples": int("${BODY_SAMPLES}"),
    "target_refined": int("${TARGET_REFINED}"),
    "temperature": float("${TEMPERATURE}"),
    "plan_max_attempt_factor": int("${PLAN_MAX_ATTEMPT_FACTOR}"),
    "plan_formula_cap": int("${PLAN_FORMULA_CAP}"),
    "plan_n_cap_fraction": float("${PLAN_N_CAP_FRACTION}"),
    "diff_steps": int("${DIFF_STEPS}"),
    "gates": {
        "plan": {
            "valid_formula_min": 0.99,
            "valid_N_min": 0.99,
            "valid_plan_min": 0.95,
            "smact_plausible_min": 0.918
        },
        "raw_body": {
            "n_parse_min": 0.99,
            "body_parse_min": 0.95,
            "graph_acceptance_min": 0.85,
            "composition_validity_min": 0.918
        },
        "refined1000": {
            "post_refine_comp_valid_min": 0.90,
            "crysllmgen_comp_valid_min": 90.0,
            "crysllmgen_struct_valid_min": 99.0,
            "crysllmgen_cov_recall_min": 90.0
        }
    }
}
Path("${NOTES_DIR}/r5d3_1000_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/lattice_geometry.py \
    crystal_dlm/llada_generation.py \
    crystal_dlm/cif_lite.py \
    crystal_dlm/crysllmgen_text.py \
    crystal_dlm/fixed_plain.py \
    scripts/sample_llada_r5_plan_compact.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5c_gate.py \
    scripts/evaluate_r5d_plan_gate.py \
    scripts/evaluate_r5b_gate.py

next_port
run_logged "${LOG_DIR}/r5d3_plan_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_plan_compact.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${PLAN_CHECKPOINT_PATH}" \
    --stats-json "${PLAN_STATS_JSON}" \
    --output-dir "${PLAN_SAMPLE_DIR}" \
    --num-samples "${PLAN_SAMPLES}" \
    --batch-size "${PLAN_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --max-attempt-factor "${PLAN_MAX_ATTEMPT_FACTOR}" \
    --formula-cap "${PLAN_FORMULA_CAP}" \
    --n-cap-fraction "${PLAN_N_CAP_FRACTION}" \
    --require-strict-smact

run_logged "${LOG_DIR}/r5d3_plan_gate1000.log" \
  python scripts/evaluate_r5d_plan_gate.py \
    --sample-metrics "${PLAN_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${PLAN_SAMPLE_DIR}/raw_generations.jsonl" \
    --min-decoded-samples "${PLAN_SAMPLES}" \
    --min-unique-formula 512 \
    --min-unique-prototype 512 \
    --output-json "${NOTES_DIR}/r5d3_plan_sample1000_gate.json" \
    --output-md "${NOTES_DIR}/r5d3_plan_sample1000_gate_report.md"
assert_gate_passed "${NOTES_DIR}/r5d3_plan_sample1000_gate.json" "R5-D3 plan sample1000"

next_port
run_logged "${LOG_DIR}/r5d3_body_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BODY_CHECKPOINT_PATH}" \
    --prompt-jsonl "${PLAN_SAMPLE_DIR}/raw_generations.jsonl" \
    --output-dir "${BODY_SAMPLE_DIR}" \
    --num-samples "${BODY_SAMPLES}" \
    --batch-size "${BODY_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

composition_raw_args=(
  python scripts/analyze_composition_validity.py
  --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl"
  --representation dynamic_v1
  --output-json "${NOTES_DIR}/r5d3_body_sample1000_composition_raw.json"
  --output-md "${NOTES_DIR}/r5d3_body_sample1000_composition_raw.md"
)
if [ -f "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt" ]; then
  composition_raw_args+=(--raw-pt "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt")
fi
run_logged "${LOG_DIR}/r5d3_body_composition_raw1000.log" "${composition_raw_args[@]}"

run_logged "${LOG_DIR}/r5d3_body_gate1000.log" \
  python scripts/evaluate_r5c_gate.py \
    --sample-metrics "${BODY_SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl" \
    --composition-summary "${NOTES_DIR}/r5d3_body_sample1000_composition_raw.json" \
    --composition-key raw_jsonl \
    --output-json "${NOTES_DIR}/r5d3_body_sample1000_gate.json" \
    --output-md "${NOTES_DIR}/r5d3_body_sample1000_gate_report.md"
assert_gate_passed "${NOTES_DIR}/r5d3_body_sample1000_gate.json" "R5-D3 body sample1000"

GRAPH_COUNT=$(python - <<PY
import torch
from pathlib import Path
path = Path("${BODY_SAMPLE_DIR}/proposal_graphs.pt")
print(len(torch.load(path, map_location="cpu")) if path.exists() else 0)
PY
)
if [ "${GRAPH_COUNT}" -lt "${TARGET_REFINED}" ]; then
  echo "Only ${GRAPH_COUNT} proposal graphs available, below TARGET_REFINED=${TARGET_REFINED}" >&2
  exit 1
fi

next_port
run_logged "${LOG_DIR}/r5d3_refined1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${BODY_SAMPLE_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps "${DIFF_STEPS}" \
    --max-proposals "${TARGET_REFINED}"

REFINED_PT=$(python - <<PY
from pathlib import Path
candidates = sorted(Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt"), key=lambda p: (p.stat().st_size, p.name), reverse=True)
candidates = [p for p in candidates if ".rank" not in p.name]
print(candidates[0] if candidates else "")
PY
)
if [ -z "${REFINED_PT}" ]; then
  echo "No merged refined pt found under ${REFINED_DIR}" >&2
  exit 1
fi

run_logged "${LOG_DIR}/r5d3_crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"

run_logged "${LOG_DIR}/r5d3_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${BODY_SAMPLE_DIR}/raw_dlm_samples.pt" \
    --refined-pt "${REFINED_PT}" \
    --raw-generations-jsonl "${BODY_SAMPLE_DIR}/raw_generations.jsonl" \
    --representation dynamic_v1 \
    --refined-world-size "${NPROC_PER_NODE}" \
    --output-json "${NOTES_DIR}/composition1000.json" \
    --output-md "${NOTES_DIR}/composition1000.md"

run_logged "${LOG_DIR}/r5d3_refined1000_gate.log" \
  python scripts/evaluate_r5b_gate.py \
    --mode refined1000 \
    --sample-metrics "${BODY_SAMPLE_DIR}/sample_metrics.json" \
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
plan = json.loads((notes / "r5d3_plan_sample1000_gate.json").read_text(encoding="utf-8"))
body = json.loads((notes / "r5d3_body_sample1000_gate.json").read_text(encoding="utf-8"))
sample = json.loads(Path("${BODY_SAMPLE_DIR}/sample_metrics.json").read_text(encoding="utf-8"))
comp = json.loads((notes / "composition1000.json").read_text(encoding="utf-8"))
crys = json.loads((notes / "crysllmgen_metrics1000.json").read_text(encoding="utf-8"))
gate = json.loads((notes / "refined1000_gate.json").read_text(encoding="utf-8"))
summary = comp.get("refined_pt") or comp.get("raw_pt") or {}
lines = [
    "# R5-D3 Plan-Conditioned Exact Body 1000 Refined Gate",
    "",
    f"- RUN_ID: ${RUN_ID}",
    f"- plan checkpoint: ${PLAN_CHECKPOINT_PATH}",
    f"- body checkpoint: ${BODY_CHECKPOINT_PATH}",
    f"- proposal graphs: ${GRAPH_COUNT}",
    f"- plan gate passed: {plan.get('passed')}",
    f"- body gate passed: {body.get('passed')}",
    f"- sample graph acceptance: {sample.get('graph_acceptance_rate')}",
    f"- refined comp_valid: {summary.get('comp_valid_rate')}",
    f"- CrysLLMGen metrics: {json.dumps(crys.get('metrics', {}), sort_keys=True)}",
    f"- refined1000 gate passed: {gate.get('passed')}",
    "",
    "## Refined Gate",
    "",
    "~~~json",
    json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True),
    "~~~",
]
(notes / "r5d3_1000_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

assert_gate_passed "${NOTES_DIR}/refined1000_gate.json" "R5-D3 refined1000"

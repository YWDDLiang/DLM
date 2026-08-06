#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260529_212834-r5c-exactlen-256}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/${PARENT_RUN_ID}/outputs/r5c_exact_sft/final}"
PROMPT_JSONL="${PROMPT_JSONL:-data/dlm_sft/mp_20_r5_exact_length/test.jsonl}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
SAMPLE_SAMPLES="${SAMPLE_SAMPLES:-1200}"
TARGET_REFINED="${TARGET_REFINED:-1000}"
TEMPERATURE="${TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
FREEZE_PLAN_COMPOSITION="${FREEZE_PLAN_COMPOSITION:-1}"
DUPLICATE_COORDINATE_MASK="${DUPLICATE_COORDINATE_MASK:-1}"
LATTICE_VOLUME_MASK="${LATTICE_VOLUME_MASK:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
SAMPLE_DIR="${OUT_DIR}/r5c_sample1000"
REFINED_DIR="${OUT_DIR}/r5c_refined1000"
SUN_DIR="${OUT_DIR}/mattergen_sun1000"
mkdir -p "${SAMPLE_DIR}" "${REFINED_DIR}" "${SUN_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

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

sample_flags=()
if [ "${FREEZE_PLAN_COMPOSITION}" = "1" ]; then
  sample_flags+=(--freeze-plan-composition)
else
  sample_flags+=(--no-freeze-plan-composition)
fi
if [ "${DUPLICATE_COORDINATE_MASK}" = "1" ]; then
  sample_flags+=(--duplicate-coordinate-mask)
else
  sample_flags+=(--no-duplicate-coordinate-mask)
fi
if [ "${LATTICE_VOLUME_MASK}" = "1" ]; then
  sample_flags+=(--lattice-volume-mask)
else
  sample_flags+=(--no-lattice-volume-mask)
fi

python - <<PY
import json
from pathlib import Path

payload = {
    "run_id": "${RUN_ID}",
    "stage": "r5c_exact_length_conditional_body_full1000_sun",
    "evaluation_scope": (
        "Conditional R5-C body diagnostic. Prompt JSONL supplies plan_state; "
        "this is not an unconditional de novo composition/planner success claim."
    ),
    "parent_run_id": "${PARENT_RUN_ID}",
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "prompt_jsonl": "${PROMPT_JSONL}",
    "model_path": "${MODEL_PATH}",
    "sample_samples": int("${SAMPLE_SAMPLES}"),
    "target_refined": int("${TARGET_REFINED}"),
    "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
    "refine_batch_size": int("${REFINE_BATCH_SIZE}"),
    "temperature": float("${TEMPERATURE}"),
    "diff_steps": int("${DIFF_STEPS}"),
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "mattergen_root": "${MATTERGEN_ROOT}",
    "reference_dataset": "${REFERENCE_DATASET}",
    "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
    "sampler_constraints": {
        "prefill_count_token": True,
        "freeze_plan_composition": "${FREEZE_PLAN_COMPOSITION}" == "1",
        "duplicate_coordinate_mask": "${DUPLICATE_COORDINATE_MASK}" == "1",
        "lattice_volume_mask": "${LATTICE_VOLUME_MASK}" == "1",
    },
}
Path("${NOTES_DIR}/r5c_full1000_run_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/lattice_geometry.py \
    crystal_dlm/llada_generation.py \
    crystal_dlm/cif_lite.py \
    crystal_dlm/crysllmgen_text.py \
    crystal_dlm/fixed_plain.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5c_gate.py \
    scripts/evaluate_r5b_gate.py \
    scripts/convert_crysllmgen_pt_to_extxyz.py \
    scripts/analyze_mattergen_sun_detailed.py

next_port
run_logged "${LOG_DIR}/r5c_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --prompt-jsonl "${PROMPT_JSONL}" \
    --output-dir "${SAMPLE_DIR}" \
    --num-samples "${SAMPLE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    "${sample_flags[@]}"

run_logged "${LOG_DIR}/r5c_composition_raw1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --representation dynamic_v1 \
    --output-json "${NOTES_DIR}/r5c_sample1000_composition_raw.json" \
    --output-md "${NOTES_DIR}/r5c_sample1000_composition_raw.md"

run_logged "${LOG_DIR}/r5c_gate1000.log" \
  python scripts/evaluate_r5c_gate.py \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --composition-summary "${NOTES_DIR}/r5c_sample1000_composition_raw.json" \
    --composition-key raw_jsonl \
    --output-json "${NOTES_DIR}/r5c_sample1000_gate.json" \
    --output-md "${NOTES_DIR}/r5c_sample1000_gate_report.md" || true

GRAPH_COUNT=$(python - <<PY
import torch
from pathlib import Path
path = Path("${SAMPLE_DIR}/proposal_graphs.pt")
print(len(torch.load(path, map_location="cpu")) if path.exists() else 0)
PY
)
echo "GRAPH_COUNT=${GRAPH_COUNT}" | tee -a "${LOG_DIR}/r5c_full1000_summary.log"
if [ "${GRAPH_COUNT}" -lt "${TARGET_REFINED}" ]; then
  echo "Only ${GRAPH_COUNT} proposal graphs available, below TARGET_REFINED=${TARGET_REFINED}" >&2
  exit 1
fi

next_port
run_logged "${LOG_DIR}/r5c_refined1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${SAMPLE_DIR}/proposal_graphs.pt" \
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

run_logged "${LOG_DIR}/r5c_crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"

run_logged "${LOG_DIR}/r5c_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --refined-pt "${REFINED_PT}" \
    --representation dynamic_v1 \
    --refined-world-size "${REFINED_WORLD_SIZE}" \
    --output-json "${NOTES_DIR}/composition1000.json" \
    --output-md "${NOTES_DIR}/composition1000.md"

run_logged "${LOG_DIR}/r5c_refined1000_gate.log" \
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
    --output-json "${NOTES_DIR}/refined1000_gate.json" || true

run_logged "${LOG_DIR}/convert_refined1000_extxyz.log" \
  python scripts/convert_crysllmgen_pt_to_extxyz.py \
    --input-pt "${REFINED_PT}" \
    --output-extxyz "${SUN_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/mattergen_sun1000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json"
  --structures-output-path "${SUN_DIR}/relaxed.extxyz"
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json"
  --relax-failures-json "${NOTES_DIR}/mattergen_sun1000_relax_failures.json"
  --unsupported-failures-json "${NOTES_DIR}/mattergen_sun1000_unsupported_failures.json"
  --metric-errors-json "${NOTES_DIR}/mattergen_sun1000_metric_errors.json"
  --relax-max-steps "${RELAX_MAX_STEPS:-500}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi

run_logged "${LOG_DIR}/mattergen_sun1000.log" \
  "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

if [ -f "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" ]; then
  run_logged "${LOG_DIR}/mattergen_sun1000_thresholds.log" \
    python scripts/analyze_mattergen_sun_detailed.py \
      --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
      --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
      --label "${RUN_ID}" \
      --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
      --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"
fi

python - <<PY
import json
from pathlib import Path

notes = Path("${NOTES_DIR}")

def read(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

payload = {
    "run_id": "${RUN_ID}",
    "evaluation_scope": "conditional_r5c_body_diagnostic_not_unconditional_de_novo",
    "graph_count": int("${GRAPH_COUNT}"),
    "sample_metrics": json.loads(Path("${SAMPLE_DIR}/sample_metrics.json").read_text(encoding="utf-8")),
    "sample_gate": read("r5c_sample1000_gate.json"),
    "composition1000": read("composition1000.json"),
    "crysllmgen_metrics1000": read("crysllmgen_metrics1000.json"),
    "refined1000_gate": read("refined1000_gate.json"),
    "mattergen_sun1000_summary": read("mattergen_sun1000_summary.json"),
    "mattergen_sun1000_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
}
(notes / "result_summary_with_sun.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

metrics = payload["sample_metrics"]
crys = payload["crysllmgen_metrics1000"].get("metrics", payload["crysllmgen_metrics1000"])
thresholds = payload["mattergen_sun1000_thresholds"]
rates = thresholds.get("rates", {}) if isinstance(thresholds, dict) else {}
lines = [
    "# R5-C Conditional Exact-Length Full1000 + S.U.N.",
    "",
    f"- RUN_ID: ${RUN_ID}",
    "- scope: conditional R5-C body diagnostic; prompt JSONL supplies plan_state, so this is not unconditional de novo.",
    f"- checkpoint: ${CHECKPOINT_PATH}",
    f"- prompt_jsonl: ${PROMPT_JSONL}",
    f"- decoded_samples: {metrics.get('decoded_samples')}",
    f"- parse_rate: {metrics.get('parse_rate')}",
    f"- graph_acceptance_rate: {metrics.get('graph_acceptance_rate')}",
    f"- proposal_graphs: ${GRAPH_COUNT}",
    f"- CrysLLMGen metrics: {json.dumps(crys, sort_keys=True)}",
    f"- strict_sun: {rates.get('strict_sun')}",
    f"- meta_sun: {rates.get('meta_sun')}",
    "",
    "## Gate",
    "",
    "~~~json",
    json.dumps(payload["refined1000_gate"], indent=2, ensure_ascii=False, sort_keys=True),
    "~~~",
]
(notes / "r5c_full1000_sun_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

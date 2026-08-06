#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
SUN_BUFFER_ACCEPTED_TIERS="${SUN_BUFFER_ACCEPTED_TIERS:-strict,meta}"
SUN_BUFFER_ACCEPTED_REASONS="${SUN_BUFFER_ACCEPTED_REASONS:-charge_neutral_pauling_valid,all_metal_shortcut}"
SUN_BUFFER_MAX_FORMULA_REPEATS="${SUN_BUFFER_MAX_FORMULA_REPEATS:-4}"
SUN_BUFFER_MAX_CHEMSYS_REPEATS="${SUN_BUFFER_MAX_CHEMSYS_REPEATS:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
SAMPLE_DIR="${RUN_DIR}/outputs/sample1000"
REFINED_DIR="${RUN_DIR}/outputs/refined1000"
SUN_DIR="${RUN_DIR}/outputs/mattergen_sun1000"
NOTES_DIR="${RUN_DIR}/notes"
BUFFER_DIR="${RUN_DIR}/outputs/self_improving_buffer"
mkdir -p "${SUN_DIR}" "${NOTES_DIR}" "${BUFFER_DIR}"

REFINED_PT="${REFINED_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt"
RAW_JSONL="${SAMPLE_DIR}/raw_generations.jsonl"
if [ ! -f "${REFINED_PT}" ]; then
  echo "Missing refined pt: ${REFINED_PT}" >&2
  exit 2
fi
if [ ! -f "${RAW_JSONL}" ]; then
  echo "Missing raw generations: ${RAW_JSONL}" >&2
  exit 2
fi

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
cfg_path = notes / "sun1000_run_config.json"
cfg_path.write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "target_graph_success": int("${TARGET_GRAPH_SUCCESS}"),
    "refined_pt": "${REFINED_PT}",
    "raw_generations_jsonl": "${RAW_JSONL}",
    "mattergen_root": "${MATTERGEN_ROOT}",
    "reference_dataset": "${REFERENCE_DATASET}",
    "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
    "accepted_tiers": "${SUN_BUFFER_ACCEPTED_TIERS}",
    "accepted_reasons": "${SUN_BUFFER_ACCEPTED_REASONS}",
    "policy": "S.U.N buffer accepts only CrysLLMGen-refined then MatterGen-relaxed strict/meta positives.",
}, indent=2, sort_keys=True) + "\n")
PY

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

"${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

python scripts/analyze_mattergen_sun_detailed.py \
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --label "${RUN_ID}" \
  --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
  --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"

python scripts/build_strict_sun_self_improving_buffer.py \
  --relaxed-extxyz "${SUN_DIR}/relaxed.extxyz" \
  --mattergen-summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --mattergen-detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --refined-pt "${REFINED_PT}" \
  --raw-generations-jsonl "${RAW_JSONL}" \
  --output-jsonl "${BUFFER_DIR}/strict_meta_sun_success.jsonl" \
  --summary-json "${NOTES_DIR}/strict_meta_sun_success_summary.json" \
  --accepted-tiers "${SUN_BUFFER_ACCEPTED_TIERS}" \
  --accepted-composition-reasons "${SUN_BUFFER_ACCEPTED_REASONS}" \
  --max-formula-repeats "${SUN_BUFFER_MAX_FORMULA_REPEATS}" \
  --max-chemsys-repeats "${SUN_BUFFER_MAX_CHEMSYS_REPEATS}"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
def read(name):
    p = notes / name
    return json.loads(p.read_text()) if p.exists() else {}
payload = {
    "run_id": "${RUN_ID}",
    "crysllmgen": read("crysllmgen_metrics1000.json"),
    "composition1000": read("composition1000.json"),
    "sun_summary": read("mattergen_sun1000_summary.json"),
    "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
    "self_improving_buffer": read("strict_meta_sun_success_summary.json"),
    "no_10000_evaluation": True,
}
(notes / "result_summary_with_sun.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

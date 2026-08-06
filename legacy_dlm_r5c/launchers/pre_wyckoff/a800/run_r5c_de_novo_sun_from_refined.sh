#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
SOURCE_RUN_DIR="runs/${SOURCE_RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
SUN_DIR="${OUT_DIR}/mattergen_sun1000"
mkdir -p "${SUN_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

REFINED_PT="${REFINED_PT:-${SOURCE_RUN_DIR}/outputs/r5c_de_novo_refined1000/dlm_refined_mp_1000.pt}"
SOURCE_SAMPLE_METRICS="${SOURCE_SAMPLE_METRICS:-${SOURCE_RUN_DIR}/outputs/r5c_de_novo_sample1000/sample_metrics.json}"
SOURCE_SAMPLE_GATE="${SOURCE_SAMPLE_GATE:-${SOURCE_RUN_DIR}/notes/r5c_de_novo_sample1000_gate.json}"
SOURCE_COMPOSITION="${SOURCE_COMPOSITION:-${SOURCE_RUN_DIR}/notes/composition1000.json}"
SOURCE_CRYS_METRICS="${SOURCE_CRYS_METRICS:-${SOURCE_RUN_DIR}/notes/crysllmgen_metrics1000.json}"
SOURCE_REFINED_GATE="${SOURCE_REFINED_GATE:-${SOURCE_RUN_DIR}/notes/refined1000_gate_acceptable.json}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:-${SOURCE_RUN_DIR}/notes/result_summary_with_sun.json}"

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
    "stage": "r5c_de_novo_sun_from_refined",
    "evaluation_scope": "S.U.N. continuation from an already refined compliant de novo R5-C DN3 run.",
    "refined_pt": "${REFINED_PT}",
    "reference_dataset": "${REFERENCE_DATASET}",
    "mattergen_root": "${MATTERGEN_ROOT}",
    "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
    "override_reason": (
        "Source run stopped before S.U.N. because the wrapper used a hard "
        "comp_valid >= 0.85 gate. User policy allows non-S.U.N. metrics within "
        "5 percentage points; source comp_valid was 0.842 / 84.2, so S.U.N. is "
        "continued without resampling or re-refinement."
    ),
    "de_novo_constraints": {
        "uses_existing_refined_structures_only": True,
        "external_plan_source": None,
        "candidate_pool_or_topk": False,
        "smact_filtered_resampling": False,
        "verifier_selection": False,
    },
    "source_files": {
        "sample_metrics": "${SOURCE_SAMPLE_METRICS}",
        "sample_gate": "${SOURCE_SAMPLE_GATE}",
        "composition": "${SOURCE_COMPOSITION}",
        "crysllmgen_metrics": "${SOURCE_CRYS_METRICS}",
        "refined_gate": "${SOURCE_REFINED_GATE}",
        "source_summary": "${SOURCE_SUMMARY}",
    },
}
Path("${NOTES_DIR}/r5c_de_novo_sun_from_refined_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if [ ! -f "${REFINED_PT}" ]; then
  echo "Missing refined pt: ${REFINED_PT}" >&2
  exit 1
fi

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/convert_crysllmgen_pt_to_extxyz.py \
    scripts/run_mattergen_sun_eval.py \
    scripts/analyze_mattergen_sun_detailed.py

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

def read_path(path_text):
    path = Path(path_text)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def read_note(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

payload = {
    "run_id": "${RUN_ID}",
    "source_run_id": "${SOURCE_RUN_ID}",
    "evaluation_scope": "r5c_composition_plan_de_novo_sun_from_refined",
    "override_policy": "Continue S.U.N. because non-S.U.N. metrics are within the user-approved 5 percentage point tolerance.",
    "source_sample_metrics": read_path("${SOURCE_SAMPLE_METRICS}"),
    "source_sample_gate": read_path("${SOURCE_SAMPLE_GATE}"),
    "source_composition1000": read_path("${SOURCE_COMPOSITION}"),
    "source_crysllmgen_metrics1000": read_path("${SOURCE_CRYS_METRICS}"),
    "source_refined1000_gate_acceptable": read_path("${SOURCE_REFINED_GATE}"),
    "mattergen_sun1000_summary": read_note("mattergen_sun1000_summary.json"),
    "mattergen_sun1000_thresholds": read_note("mattergen_sun1000_threshold_analysis.json"),
}
(notes / "result_summary_with_sun.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

sample = payload["source_sample_metrics"]
crys_payload = payload["source_crysllmgen_metrics1000"]
crys = crys_payload.get("metrics", crys_payload) if isinstance(crys_payload, dict) else {}
thresholds = payload["mattergen_sun1000_thresholds"]
rates = thresholds.get("rates", {}) if isinstance(thresholds, dict) else {}
counts = thresholds.get("counts", {}) if isinstance(thresholds, dict) else {}
summary = payload["mattergen_sun1000_summary"]

lines = [
    "# R5-C Composition-Plan De Novo S.U.N. From Refined",
    "",
    f"- RUN_ID: ${RUN_ID}",
    f"- source_RUN_ID: ${SOURCE_RUN_ID}",
    "- scope: compliant de novo S.U.N. continuation; no resampling, no re-refinement, no verifier selection.",
    "- gate note: source full1000 stopped only because comp_valid 84.2% missed the hard 85.0% gate by 0.8 pp; continued under the user-approved non-S.U.N. 5 pp tolerance.",
    f"- decoded_samples: {sample.get('decoded_samples')}",
    f"- plan_parse_rate: {sample.get('plan_parse_rate')}",
    f"- body_parse_rate: {sample.get('body_parse_rate')}",
    f"- plan_match_rate: {sample.get('plan_match_rate')}",
    f"- graph_acceptance_rate: {sample.get('graph_acceptance_rate')}",
    f"- CrysLLMGen metrics: {json.dumps(crys, sort_keys=True)}",
    f"- MatterGen submitted structures: {summary.get('num_structures') if isinstance(summary, dict) else None}",
    f"- strict_sun: {rates.get('strict_sun')}",
    f"- meta_sun: {rates.get('meta_sun')}",
    f"- counts: {json.dumps(counts, sort_keys=True)}",
    "",
]
(notes / "r5c_de_novo_sun_from_refined_report.md").write_text(
    "\n".join(lines),
    encoding="utf-8",
)
PY

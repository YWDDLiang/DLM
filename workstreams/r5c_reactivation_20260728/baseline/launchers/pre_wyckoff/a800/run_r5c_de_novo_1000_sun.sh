#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required for the de novo SFT checkpoint}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
PREREQ_GATE_JSON="${PREREQ_GATE_JSON:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
SAMPLE_SAMPLES="${SAMPLE_SAMPLES:-1200}"
TARGET_REFINED="${TARGET_REFINED:-1000}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-96}"
PLAN_STYLE="${PLAN_STYLE:-formula_text}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
SKIP_MATTERGEN_SUN="${SKIP_MATTERGEN_SUN:-0}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2 for this project." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
SAMPLE_DIR="${OUT_DIR}/r5c_de_novo_sample1000"
REFINED_DIR="${OUT_DIR}/r5c_de_novo_refined1000"
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

if [ -n "${PREREQ_GATE_JSON}" ]; then
  python - <<PY
import json
from pathlib import Path
path = Path("${PREREQ_GATE_JSON}")
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("passed_acceptable", False):
    raise SystemExit(f"Prerequisite 256 de novo gate did not pass acceptable: {path}")
PY
fi

python - <<PY
import json
from pathlib import Path

payload = {
  "run_id": "${RUN_ID}",
  "stage": "r5c_de_novo_composition_plan_full1000_sun",
  "evaluation_scope": "Compliant de novo: generated text plan is the only source for body length and composition.",
  "checkpoint_path": "${CHECKPOINT_PATH}",
  "model_path": "${MODEL_PATH}",
  "sample_samples": int("${SAMPLE_SAMPLES}"),
  "target_refined": int("${TARGET_REFINED}"),
  "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
  "refine_batch_size": int("${REFINE_BATCH_SIZE}"),
  "temperature": float("${TEMPERATURE}"),
  "plan_gen_length": int("${PLAN_GEN_LENGTH}"),
  "plan_style": "${PLAN_STYLE}",
  "diff_steps": int("${DIFF_STEPS}"),
  "skip_mattergen_sun": bool(int("${SKIP_MATTERGEN_SUN}")),
  "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
  "mattergen_root": "${MATTERGEN_ROOT}",
  "reference_dataset": "${REFERENCE_DATASET}",
  "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
  "prereq_gate_json": "${PREREQ_GATE_JSON}",
  "de_novo_constraints": {
    "external_plan_source": None,
    "candidate_pool_or_topk": False,
    "smact_filtered_resampling": False,
    "generated_plan_executes_body": True,
  },
}
Path("${NOTES_DIR}/r5c_de_novo_full1000_run_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_plan_body.py \
    crystal_dlm/r5_dynamic_length.py \
    crystal_dlm/r5_plan_state.py \
    crystal_dlm/lattice_geometry.py \
    crystal_dlm/llada_generation.py \
    crystal_dlm/cif_lite.py \
    crystal_dlm/crysllmgen_text.py \
    crystal_dlm/fixed_plain.py \
    scripts/sample_llada_r5c_plan_body.py \
    scripts/evaluate_r5c_de_novo_gate.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py \
    scripts/analyze_r5c_plan_distribution.py \
    scripts/convert_crysllmgen_pt_to_extxyz.py \
    scripts/run_mattergen_sun_eval.py \
    scripts/analyze_mattergen_sun_detailed.py

next_port
run_logged "${LOG_DIR}/r5c_de_novo_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5c_plan_body.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output-dir "${SAMPLE_DIR}" \
    --num-samples "${SAMPLE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --plan-gen-length "${PLAN_GEN_LENGTH}" \
    --plan-steps "${PLAN_GEN_LENGTH}" \
    --plan-style "${PLAN_STYLE}" \
    --freeze-plan-composition \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

gate_extra_args=()
if [ "${PLAN_STYLE}" = "formula_end_v1" ]; then
  gate_extra_args=(--enable-distribution-gates --enable-formula-end-gates)
elif [ "${PLAN_STYLE}" = "semantic_formula_v1" ]; then
  gate_extra_args=(--enable-distribution-gates --enable-semantic-gates)
fi

run_logged "${LOG_DIR}/r5c_de_novo_gate1000.log" \
  python scripts/evaluate_r5c_de_novo_gate.py \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --output-json "${NOTES_DIR}/r5c_de_novo_sample1000_gate.json" \
    --output-md "${NOTES_DIR}/r5c_de_novo_sample1000_gate.md" \
    "${gate_extra_args[@]}" || true

if [ -n "${TEACHER_JSONL:-}" ] && [ -f "${TEACHER_JSONL}" ]; then
  run_logged "${LOG_DIR}/r5c_de_novo_plan_distribution1000.log" \
    python scripts/analyze_r5c_plan_distribution.py \
      --teacher-jsonl "${TEACHER_JSONL}" \
      --generated-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
      --output-json "${NOTES_DIR}/r5c_de_novo_plan_distribution1000.json" \
      --output-md "${NOTES_DIR}/r5c_de_novo_plan_distribution1000.md"
fi

run_logged "${LOG_DIR}/r5c_de_novo_composition_raw1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --text-key body_text \
    --representation dynamic_v1 \
    --output-json "${NOTES_DIR}/r5c_de_novo_sample1000_composition_raw.json" \
    --output-md "${NOTES_DIR}/r5c_de_novo_sample1000_composition_raw.md"

GRAPH_COUNT=$(python - <<PY
import torch
from pathlib import Path
path = Path("${SAMPLE_DIR}/proposal_graphs.pt")
print(len(torch.load(path, map_location="cpu")) if path.exists() else 0)
PY
)
echo "GRAPH_COUNT=${GRAPH_COUNT}" | tee -a "${LOG_DIR}/r5c_de_novo_full1000_summary.log"
if [ "${GRAPH_COUNT}" -lt "${TARGET_REFINED}" ]; then
  echo "Only ${GRAPH_COUNT} proposal graphs available, below TARGET_REFINED=${TARGET_REFINED}" >&2
  exit 1
fi

next_port
run_logged "${LOG_DIR}/r5c_de_novo_refined1000.log" \
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

run_logged "${LOG_DIR}/r5c_de_novo_crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"

run_logged "${LOG_DIR}/r5c_de_novo_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${SAMPLE_DIR}/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
    --text-key body_text \
    --refined-pt "${REFINED_PT}" \
    --representation dynamic_v1 \
    --refined-world-size "${REFINED_WORLD_SIZE}" \
    --output-json "${NOTES_DIR}/composition1000.json" \
    --output-md "${NOTES_DIR}/composition1000.md"

run_logged "${LOG_DIR}/r5c_de_novo_refined1000_gate.log" \
  python scripts/evaluate_r5b_gate.py \
    --mode refined1000 \
    --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/composition1000.json" \
    --composition-key refined_pt \
    --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
    --min-comp-valid 0.85 \
    --min-crys-comp-valid 85.0 \
    --min-crys-struct-valid 94.0 \
    --min-crys-cov-recall 85.0 \
    --output-json "${NOTES_DIR}/refined1000_gate_acceptable.json" || true

python - <<PY
import json
from pathlib import Path

path = Path("${NOTES_DIR}/refined1000_gate_acceptable.json")
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("passed", False):
    summary = {
        "run_id": "${RUN_ID}",
        "evaluation_scope": "r5c_composition_plan_de_novo",
        "status": "stopped_before_sun",
        "reason": "refined1000 acceptable gate failed",
        "refined1000_gate_acceptable": payload,
    }
    Path("${NOTES_DIR}/result_summary_with_sun.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# R5-C Composition-Plan De Novo Full1000",
        "",
        f"- RUN_ID: ${RUN_ID}",
        "- status: stopped before S.U.N.",
        "- reason: refined1000 acceptable gate failed.",
        "",
        "```json",
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    Path("${NOTES_DIR}/r5c_de_novo_full1000_sun_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    raise SystemExit("Refined1000 acceptable gate failed; S.U.N. evaluation skipped.")
PY

if [ "${SKIP_MATTERGEN_SUN}" = "1" ]; then
  python - <<PY
import json
from pathlib import Path

notes = Path("${NOTES_DIR}")

def read(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

payload = {
    "run_id": "${RUN_ID}",
    "evaluation_scope": "r5c_composition_plan_de_novo",
    "status": "completed_without_mattergen_sun",
    "reason": "SKIP_MATTERGEN_SUN=1; A100 eval_sun is the requested S.U.N. path.",
    "refined_pt": "${REFINED_PT}",
    "graph_count": int("${GRAPH_COUNT}"),
    "sample_metrics": read("../outputs/r5c_de_novo_sample1000/sample_metrics.json"),
    "sample_gate": read("r5c_de_novo_sample1000_gate.json"),
    "plan_distribution": read("r5c_de_novo_plan_distribution1000.json"),
    "composition1000": read("composition1000.json"),
    "crysllmgen_metrics1000": read("crysllmgen_metrics1000.json"),
    "refined1000_gate_acceptable": read("refined1000_gate_acceptable.json"),
}
(notes / "result_summary_without_mattergen_sun.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = [
    "# R5-C Composition-Plan De Novo Full1000",
    "",
    f"- RUN_ID: ${RUN_ID}",
    "- status: completed without MatterGen S.U.N.",
    "- requested S.U.N. path: A100 eval_sun.py / eval_sun_resumable.py",
    f"- refined_pt: ${REFINED_PT}",
    f"- proposal_graphs: ${GRAPH_COUNT}",
    "",
    "## Refined 1000 Gate",
    "",
    "```json",
    json.dumps(payload["refined1000_gate_acceptable"], indent=2, ensure_ascii=False, sort_keys=True),
    "```",
]
(notes / "r5c_de_novo_full1000_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  exit 0
fi

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
    "evaluation_scope": "r5c_composition_plan_de_novo",
    "graph_count": int("${GRAPH_COUNT}"),
    "sample_metrics": read("../outputs/r5c_de_novo_sample1000/sample_metrics.json"),
    "sample_gate": read("r5c_de_novo_sample1000_gate.json"),
    "composition1000": read("composition1000.json"),
    "crysllmgen_metrics1000": read("crysllmgen_metrics1000.json"),
    "refined1000_gate_acceptable": read("refined1000_gate_acceptable.json"),
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
    "# R5-C Composition-Plan De Novo Full1000 + S.U.N.",
    "",
    f"- RUN_ID: ${RUN_ID}",
    "- scope: compliant de novo; body length and composition come from generated ordinary-text plan.",
    f"- checkpoint: ${CHECKPOINT_PATH}",
    f"- decoded_samples: {metrics.get('decoded_samples')}",
    f"- plan_parse_rate: {metrics.get('plan_parse_rate')}",
    f"- body_parse_rate: {metrics.get('body_parse_rate')}",
    f"- plan_match_rate: {metrics.get('plan_match_rate')}",
    f"- graph_acceptance_rate: {metrics.get('graph_acceptance_rate')}",
    f"- proposal_graphs: ${GRAPH_COUNT}",
    f"- CrysLLMGen metrics: {json.dumps(crys, sort_keys=True)}",
    f"- strict_sun: {rates.get('strict_sun')}",
    f"- meta_sun: {rates.get('meta_sun')}",
    "",
    "## Refined 1000 Gate",
    "",
    "```json",
    json.dumps(payload["refined1000_gate_acceptable"], indent=2, ensure_ascii=False, sort_keys=True),
    "```",
]
(notes / "r5c_de_novo_full1000_sun_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

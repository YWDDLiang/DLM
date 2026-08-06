#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DLM_CHECKPOINT_PATH="${DLM_CHECKPOINT_PATH:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_h1a2_rich_planner_noid}"
DATA_LIMIT="${DATA_LIMIT:-}"
GPU_COUNT="${GPU_COUNT:-2}"
PLANNER_NPROC="${PLANNER_NPROC:-${GPU_COUNT}}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
GATE_SAMPLES="${GATE_SAMPLES:-256}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-96}"
PLANNER_TEMPERATURES="${PLANNER_TEMPERATURES:-0.3 0.5 0.7}"
PLANNER_TOP_P="${PLANNER_TOP_P:-0.95}"
PLANNER_TOP_K="${PLANNER_TOP_K:-50}"
PLANNER_SEED="${PLANNER_SEED:-17}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_LR="${SFT_LR:-2e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-1.0}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
RUN_HYBRID="${RUN_HYBRID:-1}"
RUN_REFINE="${RUN_REFINE:-1}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${PLANNER_NPROC}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
  echo "GPU counts must be <=2 for this project." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((23000 + (${SLURM_JOB_ID:-0} % 20000)))}"
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

trap 'status=$?; echo "${status}" > "${NOTES_DIR}/exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/end_time.txt"; nvidia-smi > "${NOTES_DIR}/gpu_status_end.txt" 2>&1 || true; exit "${status}"' EXIT

date "+%F %T %Z" > "${NOTES_DIR}/start_time.txt"
{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-none}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort > "${NOTES_DIR}/environment.txt"

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "stage": "h1a2_rich_planner_retrain",
  "planner_model_path": "${PLANNER_MODEL_PATH}",
  "dlm_model_path": "${DLM_MODEL_PATH}",
  "dlm_checkpoint_path": "${DLM_CHECKPOINT_PATH}",
  "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
  "data_dir": "${DATA_DIR}",
  "gate_samples": int("${GATE_SAMPLES}"),
  "planner_temperatures": "${PLANNER_TEMPERATURES}".split(),
  "include_sample_id": False,
  "de_novo_constraints": {
    "gold_plan_at_sampling": False,
    "candidate_pool": False,
    "sampling_filter_or_topk": False,
    "temperature_sweep_is_method_level": True
  }
}
Path("${NOTES_DIR}/h1a2_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/h1_llm_planner.py \
    crystal_dlm/r5_plan_body.py \
    crystal_dlm/r5_dynamic_length.py \
    scripts/build_h1_llm_formula_sft_data.py \
    scripts/llama_formula_sft.py \
    scripts/sample_llama_h1_formula_plans.py \
    scripts/evaluate_h1_planner_gate.py \
    scripts/sample_llada_r5_exact_length.py \
    scripts/evaluate_h1_hybrid_gate.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py

run_logged "${LOG_DIR}/preflight_tests.log" \
  bash -lc 'python -m unittest tests.test_h1_llm_planner tests.test_r5_plan_body tests.test_r5_dynamic_length'

if [ ! -f "${DATA_DIR}/_SUCCESS" ]; then
  if [ -n "${DATA_LIMIT}" ]; then
    run_logged "${LOG_DIR}/build_h1a2_rich_planner_data.log" \
      python scripts/build_h1_llm_formula_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${DATA_DIR}" \
        --tokenizer-path "${PLANNER_MODEL_PATH}" \
        --prompt-style h1_rich_plan_v1 \
        --no-include-sample-id \
        --limit "${DATA_LIMIT}"
  else
    run_logged "${LOG_DIR}/build_h1a2_rich_planner_data.log" \
      python scripts/build_h1_llm_formula_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${DATA_DIR}" \
        --tokenizer-path "${PLANNER_MODEL_PATH}" \
        --prompt-style h1_rich_plan_v1 \
        --no-include-sample-id
  fi
fi

run_logged "${LOG_DIR}/h1a2_llama_rich_sft.log" \
  python scripts/llama_formula_sft.py \
    --model-path "${PLANNER_MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}/h1a2_llama_rich_sft" \
    --max-length "${SFT_MAX_LENGTH}" \
    --epochs "${SFT_EPOCHS}" \
    --batch-size "${SFT_BATCH_SIZE}" \
    --grad-accum "${SFT_GRAD_ACCUM}" \
    --lr "${SFT_LR}"

for temp in ${PLANNER_TEMPERATURES}; do
  label="temp${temp//./}"
  sample_dir="${OUT_DIR}/h1a2_${label}_planner256"
  next_port
  run_logged "${LOG_DIR}/h1a2_${label}_planner_sample256.log" \
    torchrun --nproc_per_node="${PLANNER_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llama_h1_formula_plans.py \
      --model-path "${PLANNER_MODEL_PATH}" \
      --checkpoint-path "${OUT_DIR}/h1a2_llama_rich_sft/final" \
      --output-dir "${sample_dir}" \
      --num-samples "${GATE_SAMPLES}" \
      --batch-size "${PLANNER_BATCH_SIZE}" \
      --max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
      --temperature "${temp}" \
      --top-p "${PLANNER_TOP_P}" \
      --top-k "${PLANNER_TOP_K}" \
      --seed "${PLANNER_SEED}" \
      --prompt-style h1_rich_plan_v1 \
      --no-include-sample-id
  run_logged "${LOG_DIR}/h1a2_${label}_planner_gate256.log" \
    python scripts/evaluate_h1_planner_gate.py \
      --sample-metrics "${sample_dir}/sample_metrics.json" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --teacher-jsonl "${DATA_DIR}/train.jsonl" \
      --output-json "${NOTES_DIR}/h1a2_${label}_planner256_gate.json" \
      --output-md "${NOTES_DIR}/h1a2_${label}_planner256_gate.md" || true
done

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
items = []
for path in sorted(notes.glob("h1a2_temp*_planner256_gate.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    comp = payload.get("distribution_comparison", {}) or {}
    cmp_metrics = comp.get("comparison", {}) if isinstance(comp, dict) else {}
    score = 0.0
    score += 1000.0 * (0 if payload.get("passed_acceptable") else 1)
    score += 100.0 * len(payload.get("acceptable_failures") or [])
    for key in ("n_tvd", "arity_tvd", "element_presence_tvd", "anion_framework_tvd", "charge_bucket_tvd", "lattice_system_tvd", "spacegroup_bucket_tvd", "volume_per_atom_bin_tvd"):
        score += float(metrics.get(key, cmp_metrics.get(key, 0.0)) or 0.0)
    items.append({"path": str(path), "stem": path.stem.replace("_planner256_gate", ""), "score": score, "passed_acceptable": bool(payload.get("passed_acceptable")), "metrics": metrics})
items.sort(key=lambda x: x["score"])
best = items[0] if items else None
(notes / "h1a2_temperature_selection.json").write_text(json.dumps({"best": best, "candidates": items}, indent=2, sort_keys=True) + "\n")
if best:
    (notes / "h1a2_best_label.txt").write_text(best["stem"] + "\n")
    (notes / "h1a2_best_acceptable.txt").write_text(("1" if best["passed_acceptable"] else "0") + "\n")
PY

BEST_LABEL="$(cat "${NOTES_DIR}/h1a2_best_label.txt")"
BEST_ACCEPTABLE="$(cat "${NOTES_DIR}/h1a2_best_acceptable.txt")"
BEST_SAMPLE_DIR="${OUT_DIR}/${BEST_LABEL}_planner256"

if [ "${RUN_HYBRID}" = "1" ] && [ "${BEST_ACCEPTABLE}" = "1" ]; then
  BODY_DIR="${OUT_DIR}/h1a2_best_hybrid_body256"
  PLAN_COUNT="$(python - <<PY
from pathlib import Path
path = Path("${BEST_SAMPLE_DIR}/plans_for_dlm.jsonl")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY
)"
  next_port
  run_logged "${LOG_DIR}/h1a2_best_hybrid_body256.log" \
    torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
      --model-path "${DLM_MODEL_PATH}" \
      --checkpoint-path "${DLM_CHECKPOINT_PATH}" \
      --prompt-jsonl "${BEST_SAMPLE_DIR}/plans_for_dlm.jsonl" \
      --output-dir "${BODY_DIR}" \
      --body-prompt-style full_plan_state \
      --num-samples "${PLAN_COUNT}" \
      --batch-size "${DLM_BATCH_SIZE}" \
      --temperature "${DLM_TEMPERATURE}" \
      --freeze-plan-composition \
      --duplicate-coordinate-mask \
      --lattice-volume-mask

  run_logged "${LOG_DIR}/h1a2_best_hybrid_gate256.log" \
    python scripts/evaluate_h1_hybrid_gate.py \
      --planner-gate-json "${NOTES_DIR}/${BEST_LABEL}_planner256_gate.json" \
      --body-sample-metrics "${BODY_DIR}/sample_metrics.json" \
      --output-json "${NOTES_DIR}/h1a2_best_hybrid256_gate.json" \
      --output-md "${NOTES_DIR}/h1a2_best_hybrid256_gate.md" || true

  if [ "${RUN_REFINE}" = "1" ] && [ -f "${BODY_DIR}/proposal_graphs.pt" ]; then
    REFINED_DIR="${OUT_DIR}/h1a2_best_refined256"
    next_port
    run_logged "${LOG_DIR}/h1a2_best_refine256.log" \
      torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${BODY_DIR}/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${REFINED_DIR}" \
        --max-proposals "${GATE_SAMPLES}" \
        --diff-steps "${DIFF_STEPS}"
    run_logged "${LOG_DIR}/h1a2_best_crysllmgen_metrics256.log" \
      python scripts/run_crysllmgen_metrics.py \
        --root-path "${REFINED_DIR}" \
        --output-json "${NOTES_DIR}/h1a2_best_crysllmgen_metrics256.json"
    REFINED_PT="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${REFINED_DIR}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt in ${REFINED_DIR}")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
    run_logged "${LOG_DIR}/h1a2_best_composition256.log" \
      python scripts/analyze_composition_validity.py \
        --raw-pt "${BODY_DIR}/raw_dlm_samples.pt" \
        --raw-generations-jsonl "${BODY_DIR}/raw_generations.jsonl" \
        --text-key text \
        --refined-pt "${REFINED_PT}" \
        --representation dynamic_v1 \
        --refined-world-size "${REFINE_NPROC}" \
        --output-json "${NOTES_DIR}/h1a2_best_composition256.json" \
        --output-md "${NOTES_DIR}/h1a2_best_composition256.md"
    run_logged "${LOG_DIR}/h1a2_best_refined_gate256.log" \
      python scripts/evaluate_r5b_gate.py \
        --mode refined1000 \
        --sample-metrics "${BODY_DIR}/sample_metrics.json" \
        --composition-summary "${NOTES_DIR}/h1a2_best_composition256.json" \
        --composition-key refined_pt \
        --crysllmgen-metrics "${NOTES_DIR}/h1a2_best_crysllmgen_metrics256.json" \
        --min-comp-valid 0.85 \
        --min-crys-comp-valid 85.0 \
        --min-crys-struct-valid 94.0 \
        --min-crys-cov-recall 85.0 \
        --output-json "${NOTES_DIR}/h1a2_best_refined256_gate.json" || true
  fi
else
  echo "H1-A2 planner did not pass acceptable gate; stopping before hybrid." | tee -a "${LOG_DIR}/h1a2_decision.log"
fi

NOTES_DIR="${NOTES_DIR}" RUN_ID="${RUN_ID}" python - <<'PY'
import json
import os
from pathlib import Path
notes = Path(os.environ["NOTES_DIR"])
run_id = os.environ["RUN_ID"]
def read(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
payload = {
    "run_id": run_id,
    "best_label": (notes / "h1a2_best_label.txt").read_text().strip() if (notes / "h1a2_best_label.txt").exists() else None,
    "temperature_selection": read("h1a2_temperature_selection.json"),
    "hybrid_gate": read("h1a2_best_hybrid256_gate.json"),
    "refined_gate": read("h1a2_best_refined256_gate.json"),
    "crysllmgen_metrics256": read("h1a2_best_crysllmgen_metrics256.json"),
}
(notes / "h1a2_result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = ["# H1-A2 Rich Planner Retrain Report", "", f"- RUN_ID: {run_id}", f"- best_label: {payload['best_label']}", ""]
lines += ["## Summary JSON", "", "```json", json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), "```", ""]
(notes / "h1a2_rich_planner_retrain_report.md").write_text("\n".join(lines), encoding="utf-8")
PY

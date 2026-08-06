#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260527_chemplan_bridge_r3}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PREV_RUN_ID="${PREV_RUN_ID:-20260527_semalign_selfimprove_r2}"
PREV_BEST_CHECKPOINT="${PREV_BEST_CHECKPOINT:-runs/${PREV_RUN_ID}/outputs/stage_b/final}"
BASE_INPUT_DATA_DIR="${BASE_INPUT_DATA_DIR:-data/dlm_sft/mp_20_sun_self_improve_weighted_${PREV_RUN_ID}}"
FALLBACK_INPUT_DATA_DIR="${FALLBACK_INPUT_DATA_DIR:-data/dlm_sft/mp_20}"
CHEM_PLAN_DATA_DIR="${CHEM_PLAN_DATA_DIR:-data/dlm_sft/mp_20_chem_plan_${RUN_ID}}"
PLAN_ROW_FRACTION="${PLAN_ROW_FRACTION:-0.35}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PLAN_TEMPERATURE="${PLAN_TEMPERATURE:-0.7}"
PLAN_GEN_LENGTH="${PLAN_GEN_LENGTH:-96}"
PLAN_BLOCK_LENGTH="${PLAN_BLOCK_LENGTH:-16}"
PLAN_STEPS="${PLAN_STEPS:-96}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-384}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT:-4}}"
MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.88}"
MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.40}"
MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
MAX_PBC_DUPLICATE_256="${MAX_PBC_DUPLICATE_256:-0.0}"
FORCE_1000="${FORCE_1000:-0}"
RUN_STAGE_B="${RUN_STAGE_B:-0}"
STAGE_A_LR="${STAGE_A_LR:-2e-6}"
STAGE_A_WARMUP_STEPS="${STAGE_A_WARMUP_STEPS:-20}"
STAGE_B_LR="${STAGE_B_LR:-8e-7}"
STAGE_B_WARMUP_STEPS="${STAGE_B_WARMUP_STEPS:-20}"
WEIGHTED_SAMPLING="${WEIGHTED_SAMPLING:-1}"
WEIGHTED_SAMPLING_POWER="${WEIGHTED_SAMPLING_POWER:-1.0}"
SAMPLE_WEIGHT_MULTIPLIERS="${SAMPLE_WEIGHT_MULTIPLIERS:-all_metal=0.35,single_element=0.02,invalid=0.15,selection_role:self_improving_repeat=1.3,selection_role:chemical_plan=0.35}"
IGNORE_JSONL_SAMPLE_WEIGHT="${IGNORE_JSONL_SAMPLE_WEIGHT:-0}"
ATOM_COUNT_LOSS_WEIGHT="${ATOM_COUNT_LOSS_WEIGHT:-3.0}"
SLOT_MARKER_LOSS_WEIGHT="${SLOT_MARKER_LOSS_WEIGHT:-0.25}"
EMPTY_SLOT_LOSS_WEIGHT="${EMPTY_SLOT_LOSS_WEIGHT:-0.20}"
NONEMPTY_SLOT_LOSS_WEIGHT="${NONEMPTY_SLOT_LOSS_WEIGHT:-2.5}"
LATE_NONEMPTY_SLOT_LOSS_WEIGHT="${LATE_NONEMPTY_SLOT_LOSS_WEIGHT:-4.0}"
COORDINATE_LOSS_WEIGHT="${COORDINATE_LOSS_WEIGHT:-1.2}"
PAD_COORDINATE_LOSS_WEIGHT="${PAD_COORDINATE_LOSS_WEIGHT:-0.10}"
ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT:-0.08}"
ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT:-1.0}"
MASTER_PORT_BASE="${MASTER_PORT:-$((23000 + (${SLURM_JOB_ID:-0} % 25000)))}"

cd "${PROJECT_ROOT}"
if [ ! -d "${PREV_BEST_CHECKPOINT}" ]; then
  echo "PREV_BEST_CHECKPOINT does not exist: ${PREV_BEST_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -f "${PREV_BEST_CHECKPOINT}/adapter_config.json" ] && [ ! -f "${PREV_BEST_CHECKPOINT}/config.json" ]; then
  echo "PREV_BEST_CHECKPOINT is not a recognizable checkpoint directory: ${PREV_BEST_CHECKPOINT}" >&2
  exit 2
fi
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
REPORT_DIR="reports"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}" "${REPORT_DIR}"

PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((MASTER_PORT_BASE + PORT_OFFSET))
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

cleanup_weight_files_except() {
  local keep_a="${1:-}"
  local keep_b="${2:-}"
  python - <<PY
from pathlib import Path
import json

root = Path("${OUT_DIR}")
keep = {Path(p).resolve() for p in ["${keep_a}", "${keep_b}"] if p}
patterns = ("*.safetensors", "*.bin", "*.pt")
deleted = []
for pattern in patterns:
    for path in root.rglob(pattern):
        resolved = path.resolve()
        try:
            should_keep = any(resolved.is_relative_to(k) for k in keep)
        except AttributeError:
            should_keep = any(str(resolved).startswith(str(k) + "/") or resolved == k for k in keep)
        if should_keep:
            continue
        deleted.append(str(path))
        path.unlink()
Path("${NOTES_DIR}/weight_cleanup_last.json").write_text(
    json.dumps({"kept": [str(k) for k in keep], "deleted": deleted}, indent=2, sort_keys=True) + "\n"
)
print(f"deleted_weight_files={len(deleted)}")
PY
}

if [ ! -f "${BASE_INPUT_DATA_DIR}/train.jsonl" ]; then
  echo "BASE_INPUT_DATA_DIR=${BASE_INPUT_DATA_DIR} missing; falling back to ${FALLBACK_INPUT_DATA_DIR}."
  BASE_INPUT_DATA_DIR="${FALLBACK_INPUT_DATA_DIR}"
fi
if [ ! -f "${BASE_INPUT_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_INPUT_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "purpose": "R3 chemical-plan bridge: natural-language chemistry plan + fixed-slot constrained structure generation.",
  "model_path": "${MODEL_PATH}",
  "prev_best_checkpoint": "${PREV_BEST_CHECKPOINT}",
  "base_input_data_dir": "${BASE_INPUT_DATA_DIR}",
  "chem_plan_data_dir": "${CHEM_PLAN_DATA_DIR}",
  "plan_row_fraction": float("${PLAN_ROW_FRACTION}"),
  "temperature": float("${TEMPERATURE}"),
  "plan_temperature": float("${PLAN_TEMPERATURE}"),
  "plan_gen_length": int("${PLAN_GEN_LENGTH}"),
  "plan_block_length": int("${PLAN_BLOCK_LENGTH}"),
  "plan_steps": int("${PLAN_STEPS}"),
  "nproc_per_node": int("${NPROC_PER_NODE}"),
  "sft_batch_size": int("${SFT_BATCH_SIZE}"),
  "sft_grad_accum": int("${SFT_GRAD_ACCUM}"),
  "sft_max_length": int("${SFT_MAX_LENGTH}"),
  "stage_a_lr": "${STAGE_A_LR}",
  "stage_b_lr": "${STAGE_B_LR}",
  "run_stage_b": bool(int("${RUN_STAGE_B}")),
  "force_1000": bool(int("${FORCE_1000}")),
  "sample_weight_multipliers": "${SAMPLE_WEIGHT_MULTIPLIERS}",
  "no_10000_evaluation": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [ "${REBUILD_CHEM_PLAN_DATA:-0}" = "1" ]; then
  rm -rf "${CHEM_PLAN_DATA_DIR}"
fi
if [ ! -f "${CHEM_PLAN_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_chemical_plan_data.log" \
    python scripts/build_fixed_slot_chemical_plan_sft_data.py \
      --input-dir "${BASE_INPUT_DATA_DIR}" \
      --output-dir "${CHEM_PLAN_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --plan-row-fraction "${PLAN_ROW_FRACTION}"
fi
cp "${CHEM_PLAN_DATA_DIR}/chemical_plan_summary.json" "${NOTES_DIR}/input_chemical_plan_summary.json" || true
cp "${CHEM_PLAN_DATA_DIR}/stats.json" "${NOTES_DIR}/input_chemical_plan_stats.json" || true

train_stage() {
  local stage_name="$1"
  local output_dir="$2"
  local lr="$3"
  local warmup_steps="$4"
  local checkpoint_path="$5"
  next_port
  local cmd=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/llada_sft.py
      --model-path "${MODEL_PATH}"
      --checkpoint-path "${checkpoint_path}"
      --data-dir "${CHEM_PLAN_DATA_DIR}"
      --output-dir "${output_dir}"
      --max-length "${SFT_MAX_LENGTH}"
      --epochs 1
      --batch-size "${SFT_BATCH_SIZE}"
      --grad-accum "${SFT_GRAD_ACCUM}"
      --lr "${lr}"
      --lr-scheduler cosine
      --warmup-steps "${warmup_steps}"
      --min-lr-ratio 0.2
      --atom-count-loss-weight "${ATOM_COUNT_LOSS_WEIGHT}"
      --slot-marker-loss-weight "${SLOT_MARKER_LOSS_WEIGHT}"
      --empty-slot-loss-weight "${EMPTY_SLOT_LOSS_WEIGHT}"
      --nonempty-slot-loss-weight "${NONEMPTY_SLOT_LOSS_WEIGHT}"
      --late-nonempty-slot-loss-weight "${LATE_NONEMPTY_SLOT_LOSS_WEIGHT}"
      --coordinate-loss-weight "${COORDINATE_LOSS_WEIGHT}"
      --pad-coordinate-loss-weight "${PAD_COORDINATE_LOSS_WEIGHT}"
      --semantic-init-element-tokens
      --element-token-alignment-loss-weight "${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT}"
      --element-token-alignment-output-weight "${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT}"
      --train-prefill-slot-tokens
      --logging-steps 20
      --eval-steps 424
      --save-steps 848
      --position-diagnostics-steps 424
  )
  if [ "${WEIGHTED_SAMPLING}" = "1" ]; then
    cmd+=(--weighted-sampling --weighted-sampling-power "${WEIGHTED_SAMPLING_POWER}")
    if [ -n "${SAMPLE_WEIGHT_MULTIPLIERS}" ]; then
      cmd+=(--sample-weight-multipliers "${SAMPLE_WEIGHT_MULTIPLIERS}")
    fi
    if [ "${IGNORE_JSONL_SAMPLE_WEIGHT}" = "1" ]; then
      cmd+=(--ignore-jsonl-sample-weight)
    fi
  fi
  run_logged "${LOG_DIR}/${stage_name}_train.log" "${cmd[@]}"
}

summarize_training_log() {
  local stage_name="$1"
  local output_dir="$2"
  local candidate_notes="${NOTES_DIR}/${stage_name}"
  mkdir -p "${candidate_notes}"
  python - <<PY
import json
from pathlib import Path

log_path = Path("${output_dir}") / "training_log.jsonl"
events = []
if log_path.exists():
    for line in log_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
evals = [e for e in events if e.get("event") == "eval"]
diagnostics = [e for e in events if e.get("event") == "position_diagnostics"]
trains = [e for e in events if e.get("event") == "train"]
starts = [e for e in events if e.get("event") == "start"]
latest_diag = diagnostics[-1] if diagnostics else {}
payload = {
    "stage": "${stage_name}",
    "training_log": str(log_path),
    "latest_train": trains[-1] if trains else {},
    "latest_eval": evals[-1] if evals else {},
    "latest_group_ce": latest_diag.get("group_ce", {}),
    "latest_group_ce_summary": latest_diag.get("group_ce_summary", {}),
    "weighted_sampling": starts[0].get("weighted_sampling", {}) if starts else {},
}
Path("${candidate_notes}/training_diagnostics_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

summarize_plans() {
  local raw_jsonl="$1"
  local output_json="$2"
  local output_md="$3"
  python - <<PY
import json
from collections import Counter
from pathlib import Path

raw = Path("${raw_jsonl}")
plans = []
if raw.exists():
    with raw.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            plan = str(row.get("plan") or "")
            if plan:
                plans.append(plan)
keys = Counter()
reasons = Counter()
for plan in plans:
    for label in ("formula:", "composition:", "composition_reason:", "chemistry:", "stability_hint:", "geometry_hint:", "crystal tokens:"):
        if label in plan:
            keys[label] += 1
    for line in plan.splitlines():
        if line.startswith("composition_reason:"):
            reasons[line.split(":", 1)[1].strip()] += 1
payload = {
    "plan_count": len(plans),
    "field_presence": dict(keys),
    "composition_reasons": dict(reasons.most_common()),
    "examples": plans[:5],
}
Path("${output_json}").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
md = ["# Chemical Plan Summary", "", f"- plan_count: {len(plans)}", ""]
md.append("## Field presence")
for key, value in keys.most_common():
    md.append(f"- {key} {value}")
md.append("")
md.append("## Composition reasons")
for key, value in reasons.most_common(20):
    md.append(f"- {key}: {value}")
md.append("")
md.append("## Examples")
for idx, plan in enumerate(plans[:5], 1):
    md.append(f"### Example {idx}")
    md.append("```text")
    md.append(plan)
    md.append("```")
Path("${output_md}").write_text("\n".join(md) + "\n", encoding="utf-8")
PY
}

smoke256() {
  local candidate="$1"
  local checkpoint="$2"
  local sample_dir="${OUT_DIR}/${candidate}_sample256"
  local candidate_notes="${NOTES_DIR}/${candidate}"
  mkdir -p "${sample_dir}" "${candidate_notes}"
  next_port
  run_logged "${LOG_DIR}/${candidate}_sample256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_chemical_plan_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${sample_dir}" \
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
      --lattice-volume-mask
  summarize_plans "${sample_dir}/raw_generations.jsonl" "${candidate_notes}/plan_summary.json" "${candidate_notes}/plan_summary.md"
  python scripts/analyze_sample_outputs.py \
    --input-jsonl "${sample_dir}/raw_generations.jsonl" \
    --failure-jsonl "${sample_dir}/failure_cases.jsonl" \
    --output-json "${candidate_notes}/sample256_distribution.json" \
    --output-md "${candidate_notes}/sample256_distribution.md"
  python scripts/analyze_composition_validity.py \
    --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${candidate_notes}/sample256_composition.json" \
    --output-md "${candidate_notes}/sample256_composition.md"
  python scripts/analyze_composition_failure_modes.py \
    --raw-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${candidate_notes}/sample256_failure_modes.json" \
    --output-md "${candidate_notes}/sample256_failure_modes.md"
  python scripts/evaluate_mp20_candidate_gate.py \
    --mode smoke256 \
    --sample-metrics "${sample_dir}/sample_metrics.json" \
    --composition-summary "${candidate_notes}/sample256_composition.json" \
    --composition-key raw_jsonl \
    --min-parse-rate 0.98 \
    --min-graph-acceptance 0.95 \
    --min-comp-valid "${MIN_COMP_VALID_256}" \
    --min-strict-valid "${MIN_STRICT_VALID_256}" \
    --max-single-element "${MAX_SINGLE_ELEMENT_256}" \
    --max-pbc-duplicate "${MAX_PBC_DUPLICATE_256}" \
    --output-json "${candidate_notes}/sample256_gate.json"
}

stage_passed() {
  local name="$1"
  python - <<PY
import json
from pathlib import Path
path = Path("${NOTES_DIR}") / "${name}" / "sample256_gate.json"
print("1" if path.exists() and json.loads(path.read_text()).get("passed") else "0")
PY
}

STAGE_A_DIR="${OUT_DIR}/stage_a"
train_stage "stage_a" "${STAGE_A_DIR}" "${STAGE_A_LR}" "${STAGE_A_WARMUP_STEPS}" "${PREV_BEST_CHECKPOINT}"
summarize_training_log "stage_a" "${STAGE_A_DIR}" | tee "${LOG_DIR}/stage_a_training_diagnostics_summary.log"
smoke256 "stage_a" "${STAGE_A_DIR}/final"

if [ "${STOP_AFTER_256:-0}" = "1" ]; then
  echo "STOP_AFTER_256=1; stopping chemical-plan diagnostic after stage_a 256 smoke."
  cleanup_weight_files_except "${STAGE_A_DIR}/final"
  exit 0
fi

if [ "$(stage_passed stage_a)" != "1" ]; then
  echo "Stage A failed 256 gate; stopping before 1000 unless FORCE_1000=1."
  if [ "${FORCE_1000}" != "1" ]; then
    cleanup_weight_files_except "${STAGE_A_DIR}/final"
    exit 0
  fi
fi

BEST_CHECKPOINT="${STAGE_A_DIR}/final"
BEST_NAME="stage_a"
LATEST_CHECKPOINT="${STAGE_A_DIR}/final"

if [ "${RUN_STAGE_B}" = "1" ]; then
  STAGE_B_DIR="${OUT_DIR}/stage_b"
  train_stage "stage_b" "${STAGE_B_DIR}" "${STAGE_B_LR}" "${STAGE_B_WARMUP_STEPS}" "${STAGE_A_DIR}/final"
  summarize_training_log "stage_b" "${STAGE_B_DIR}" | tee "${LOG_DIR}/stage_b_training_diagnostics_summary.log"
  smoke256 "stage_b" "${STAGE_B_DIR}/final"
  BEST_CHECKPOINT="$(python - <<PY
import json
from pathlib import Path
candidates = []
for name, ckpt in [("stage_a", "${STAGE_A_DIR}/final"), ("stage_b", "${STAGE_B_DIR}/final")]:
    gate_path = Path("${NOTES_DIR}") / name / "sample256_gate.json"
    failure_path = Path("${NOTES_DIR}") / name / "sample256_failure_modes.json"
    gate = json.loads(gate_path.read_text())
    metrics = gate.get("metrics", {})
    reasons = json.loads(failure_path.read_text()).get("reason_counts", {}) if failure_path.exists() else {}
    candidates.append({
        "name": name,
        "checkpoint": ckpt,
        "passed": bool(gate.get("passed")),
        "comp_valid": float(metrics.get("comp_valid") or 0.0),
        "strict_valid": float(metrics.get("strict_valid") or 0.0),
        "single_element": float(metrics.get("single_element") or 0.0),
        "charge_fail": int(reasons.get("charge_neutrality_fail", 999999)),
    })
candidates.sort(key=lambda x: (x["passed"], x["comp_valid"], x["strict_valid"], -x["single_element"], -x["charge_fail"]), reverse=True)
Path("${NOTES_DIR}/candidate_selection.json").write_text(json.dumps({"candidates": candidates, "best": candidates[0]}, indent=2, sort_keys=True) + "\n")
print(candidates[0]["checkpoint"])
PY
)"
  BEST_NAME="$(python - <<PY
import json
from pathlib import Path
print(json.loads(Path("${NOTES_DIR}/candidate_selection.json").read_text())["best"]["name"])
PY
)"
  LATEST_CHECKPOINT="${STAGE_B_DIR}/final"
else
  python - <<PY
import json
from pathlib import Path
gate = json.loads((Path("${NOTES_DIR}") / "stage_a" / "sample256_gate.json").read_text())
payload = {"candidates": [{"name": "stage_a", "checkpoint": "${STAGE_A_DIR}/final", "passed": gate.get("passed"), "metrics": gate.get("metrics", {})}], "best": {"name": "stage_a", "checkpoint": "${STAGE_A_DIR}/final", "passed": gate.get("passed"), "metrics": gate.get("metrics", {})}}
Path("${NOTES_DIR}/candidate_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
fi

echo "BEST_NAME=${BEST_NAME}"
echo "BEST_CHECKPOINT=${BEST_CHECKPOINT}"
cleanup_weight_files_except "${BEST_CHECKPOINT}" "${LATEST_CHECKPOINT}"

BEST_PASSED="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/candidate_selection.json").read_text())
print("1" if payload.get("best", {}).get("passed") else "0")
PY
)"
if [ "${BEST_PASSED}" != "1" ] && [ "${FORCE_1000}" != "1" ]; then
  echo "Best candidate failed 256 gate; stopping before 1000/refinement/SUN."
  exit 0
fi

SAMPLE1000_DIR="${OUT_DIR}/sample1000"
REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/mattergen_sun1000"
BUFFER_DIR="data/dlm_sft/mp_20_strict_sun_buffer_${RUN_ID}"
mkdir -p "${SAMPLE1000_DIR}" "${REFINED1000_DIR}" "${SUN1000_DIR}" "${BUFFER_DIR}"

next_port
run_logged "${LOG_DIR}/sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_chemical_plan_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --output-dir "${SAMPLE1000_DIR}" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --num-samples "${MAX_ATTEMPTS}" \
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
    --lattice-volume-mask

summarize_plans "${SAMPLE1000_DIR}/raw_generations.jsonl" "${NOTES_DIR}/sample1000_plan_summary.json" "${NOTES_DIR}/sample1000_plan_summary.md"
python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE1000_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/sample1000_distribution.json" \
  --output-md "${NOTES_DIR}/sample1000_distribution.md"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample1000_composition_raw.json" \
  --output-md "${NOTES_DIR}/sample1000_composition_raw.md"
python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample1000_failure_modes_raw.json" \
  --output-md "${NOTES_DIR}/sample1000_failure_modes_raw.md"

TARGET_REACHED="$(python - <<PY
import json
from pathlib import Path
print("1" if json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()).get("target_reached") else "0")
PY
)"
if [ "${TARGET_REACHED}" != "1" ]; then
  echo "1000 graph-valid target not reached; stopping before refinement/SUN."
  exit 0
fi

next_port
run_logged "${LOG_DIR}/refine1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${SAMPLE1000_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED1000_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps "${DIFF_STEPS}" \
    --max-proposals "${TARGET_GRAPH_SUCCESS}"

REFINED_PT="${REFINED1000_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt"
python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED1000_DIR}" \
  --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --refined-pt "${REFINED_PT}" \
  --refined-world-size 2 \
  --output-json "${NOTES_DIR}/composition1000.json" \
  --output-md "${NOTES_DIR}/composition1000.md"
python scripts/evaluate_mp20_candidate_gate.py \
  --mode refined1000 \
  --sample-metrics "${SAMPLE1000_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/composition1000.json" \
  --composition-key refined_pt \
  --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
  --max-single-element 0.10 \
  --max-pbc-duplicate 0.0 \
  --output-json "${NOTES_DIR}/refined1000_gate.json" || true

python scripts/convert_crysllmgen_pt_to_extxyz.py \
  --input-pt "${REFINED_PT}" \
  --output-extxyz "${SUN1000_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN1000_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/mattergen_sun1000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json"
  --structures-output-path "${SUN1000_DIR}/relaxed.extxyz"
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

python scripts/analyze_mattergen_sun_detailed.py \
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --label "${RUN_ID}" \
  --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
  --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"

python scripts/build_strict_sun_self_improving_buffer.py \
  --relaxed-extxyz "${SUN1000_DIR}/relaxed.extxyz" \
  --mattergen-summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --mattergen-detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --refined-pt "${REFINED_PT}" \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-jsonl "${BUFFER_DIR}/strict_sun_success.jsonl" \
  --summary-json "${BUFFER_DIR}/strict_sun_success_summary.json" \
  --accepted-tiers "strict,meta" \
  --accepted-composition-reasons "charge_neutral_pauling_valid,all_metal_shortcut" \
  --max-formula-repeats 4 \
  --max-chemsys-repeats 32
cp "${BUFFER_DIR}/strict_sun_success_summary.json" "${NOTES_DIR}/strict_sun_success_summary.json" || true

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
def read(name):
    path = notes / name
    return json.loads(path.read_text()) if path.exists() else {}
payload = {
  "run_id": "${RUN_ID}",
  "best_name": "${BEST_NAME}",
  "best_checkpoint": "${BEST_CHECKPOINT}",
  "candidate_selection": read("candidate_selection.json"),
  "sample1000": json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()),
  "plan1000": read("sample1000_plan_summary.json"),
  "crysllmgen": read("crysllmgen_metrics1000.json"),
  "composition1000": read("composition1000.json"),
  "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
  "strict_sun_buffer": read("strict_sun_success_summary.json"),
  "no_10000_evaluation": True,
}
(notes / "result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cp "${NOTES_DIR}/result_summary.json" "${REPORT_DIR}/${RUN_ID}_result_summary.json"

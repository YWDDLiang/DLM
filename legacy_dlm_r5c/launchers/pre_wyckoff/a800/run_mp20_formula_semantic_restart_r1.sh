#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260528_formula_semantic_restart_r1}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
SEMANTIC_DATA_DIR="${SEMANTIC_DATA_DIR:-data/dlm_sft/mp_20_formula_semantic_r1}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"
SEMANTIC_FRACTION="${SEMANTIC_FRACTION:-1.0}"
SEMANTIC_WEIGHT_MULTIPLIER="${SEMANTIC_WEIGHT_MULTIPLIER:-1.0}"
SEMANTIC_INCLUDE_REASONS="${SEMANTIC_INCLUDE_REASONS:-}"
SEMANTIC_EXCLUDE_REASONS="${SEMANTIC_EXCLUDE_REASONS:-}"
SEMANTIC_MIN_ELEMENTS="${SEMANTIC_MIN_ELEMENTS:-0}"
SEMANTIC_MAX_ELEMENTS="${SEMANTIC_MAX_ELEMENTS:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TEMPERATURE_COMPARE="${TEMPERATURE_COMPARE:-1.0}"
RUN_TEMPERATURE_COMPARE="${RUN_TEMPERATURE_COMPARE:-0}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT:-2}}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-384}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-3}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-2}"
STAGE_A_LR="${STAGE_A_LR:-5e-5}"
STAGE_B_LR="${STAGE_B_LR:-1e-5}"
STAGE_A_WARMUP_STEPS="${STAGE_A_WARMUP_STEPS:-100}"
STAGE_B_WARMUP_STEPS="${STAGE_B_WARMUP_STEPS:-50}"
ATOM_COUNT_LOSS_WEIGHT="${ATOM_COUNT_LOSS_WEIGHT:-3.0}"
SLOT_MARKER_LOSS_WEIGHT="${SLOT_MARKER_LOSS_WEIGHT:-0.25}"
EMPTY_SLOT_LOSS_WEIGHT="${EMPTY_SLOT_LOSS_WEIGHT:-0.5}"
NONEMPTY_SLOT_LOSS_WEIGHT="${NONEMPTY_SLOT_LOSS_WEIGHT:-2.0}"
LATE_NONEMPTY_SLOT_LOSS_WEIGHT="${LATE_NONEMPTY_SLOT_LOSS_WEIGHT:-4.0}"
COORDINATE_LOSS_WEIGHT="${COORDINATE_LOSS_WEIGHT:-1.0}"
PAD_COORDINATE_LOSS_WEIGHT="${PAD_COORDINATE_LOSS_WEIGHT:-0.2}"
ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT:-0.05}"
ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT:-1.0}"
TRAIN_WEIGHTED_SAMPLING="${TRAIN_WEIGHTED_SAMPLING:-0}"
WEIGHTED_SAMPLING_POWER="${WEIGHTED_SAMPLING_POWER:-1.0}"
SAMPLE_WEIGHT_MULTIPLIERS="${SAMPLE_WEIGHT_MULTIPLIERS:-}"
MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.88}"
MIN_PARSE_RATE_256="${MIN_PARSE_RATE_256:-1.0}"
MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.40}"
MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
MAX_PBC_DUPLICATE_256="${MAX_PBC_DUPLICATE_256:-0.0}"
FORCE_STAGE_B="${FORCE_STAGE_B:-0}"
FORCE_1000="${FORCE_1000:-0}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
RELAX_MAX_STEPS="${RELAX_MAX_STEPS:-500}"
MAX_NATOMS_PER_BATCH="${MAX_NATOMS_PER_BATCH:-512}"
MASTER_PORT_BASE="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"

if [ "${NPROC_PER_NODE}" -gt 2 ]; then
  echo "NPROC_PER_NODE/GPU_COUNT must be <= 2." >&2
  exit 2
fi
if [ "${TEMPERATURE}" != "0.7" ] && [ "${TEMPERATURE}" != "1.0" ]; then
  echo "TEMPERATURE must be 0.7 or 1.0." >&2
  exit 2
fi
if [ "${TEMPERATURE_COMPARE}" != "0.7" ] && [ "${TEMPERATURE_COMPARE}" != "1.0" ]; then
  echo "TEMPERATURE_COMPARE must be 0.7 or 1.0." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
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

cleanup_weight_dirs_except() {
  local keep_a="${1:-}"
  local keep_b="${2:-}"
  python - <<PY
import json
import shutil
from pathlib import Path

root = Path("${OUT_DIR}").resolve()
keep = {Path(p).resolve() for p in ["${keep_a}", "${keep_b}"] if p}
weight_names = {"adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin"}

def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def has_direct_weight(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_file():
            continue
        if child.name in weight_names or (child.name.startswith("pytorch_model-") and child.name.endswith(".bin")):
            return True
        if child.name.endswith(".safetensors"):
            return True
    return False

deleted = []
if root.exists():
    for path in sorted(root.rglob("*")):
        if not path.is_dir() or not has_direct_weight(path):
            continue
        resolved = path.resolve()
        if any(resolved == item or is_inside(resolved, item) for item in keep):
            continue
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path)
        deleted.append({"path": str(path), "bytes": int(size)})
payload = {"kept": [str(item) for item in sorted(keep)], "deleted": deleted}
Path("${NOTES_DIR}/weight_cleanup_last.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "model_path": "${MODEL_PATH}",
  "base_data_dir": "${BASE_DATA_DIR}",
  "semantic_data_dir": "${SEMANTIC_DATA_DIR}",
  "semantic_fraction": float("${SEMANTIC_FRACTION}"),
  "semantic_weight_multiplier": float("${SEMANTIC_WEIGHT_MULTIPLIER}"),
  "semantic_include_reasons": "${SEMANTIC_INCLUDE_REASONS}",
  "semantic_exclude_reasons": "${SEMANTIC_EXCLUDE_REASONS}",
  "semantic_min_elements": int("${SEMANTIC_MIN_ELEMENTS}"),
  "semantic_max_elements": int("${SEMANTIC_MAX_ELEMENTS}"),
  "temperature": float("${TEMPERATURE}"),
  "temperature_compare": float("${TEMPERATURE_COMPARE}"),
  "run_temperature_compare": bool(int("${RUN_TEMPERATURE_COMPARE}")),
  "generation_schedule": "${GENERATION_SCHEDULE}",
  "nproc_per_node": int("${NPROC_PER_NODE}"),
  "sft_batch_size": int("${SFT_BATCH_SIZE}"),
  "sft_grad_accum": int("${SFT_GRAD_ACCUM}"),
  "sft_max_length": int("${SFT_MAX_LENGTH}"),
  "stage_a_epochs": int("${STAGE_A_EPOCHS}"),
  "stage_b_epochs": int("${STAGE_B_EPOCHS}"),
  "stage_a_lr": "${STAGE_A_LR}",
  "stage_b_lr": "${STAGE_B_LR}",
  "train_weighted_sampling": bool(int("${TRAIN_WEIGHTED_SAMPLING}")),
  "weighted_sampling_power": float("${WEIGHTED_SAMPLING_POWER}"),
  "sample_weight_multipliers": "${SAMPLE_WEIGHT_MULTIPLIERS}",
  "min_parse_rate_256": float("${MIN_PARSE_RATE_256}"),
  "element_token_alignment_loss_weight": float("${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT}"),
  "no_online_plan_generation": True,
  "no_10000_evaluation": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_tests.log" \
  python -m unittest \
    tests.test_fixed_slot_formula_semantic_sft_data \
    tests.test_llada_sft_weights \
    tests.test_llada_generation_masks \
    tests.test_composition_validity

python -m py_compile \
  scripts/build_fixed_slot_formula_semantic_sft_data.py \
  scripts/llada_sft.py \
  scripts/sample_llada_crystals.py \
  scripts/analyze_composition_failure_modes.py \
  scripts/run_mattergen_sun_eval.py

if [ ! -f "${BASE_DATA_DIR}/train.jsonl" ] || [ ! -f "${BASE_DATA_DIR}/vocab_tokens.txt" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

if [ "${FORCE_REBUILD_DATA}" = "1" ] || [ ! -f "${SEMANTIC_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_formula_semantic_data.log" \
    python scripts/build_fixed_slot_formula_semantic_sft_data.py \
      --input-dir "${BASE_DATA_DIR}" \
      --output-dir "${SEMANTIC_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --semantic-fraction "${SEMANTIC_FRACTION}" \
      --semantic-weight-multiplier "${SEMANTIC_WEIGHT_MULTIPLIER}" \
      --semantic-include-reasons "${SEMANTIC_INCLUDE_REASONS}" \
      --semantic-exclude-reasons "${SEMANTIC_EXCLUDE_REASONS}" \
      --semantic-min-elements "${SEMANTIC_MIN_ELEMENTS}" \
      --semantic-max-elements "${SEMANTIC_MAX_ELEMENTS}"
fi

SMOKE_DIR="${OUT_DIR}/stage0_smoke32"
run_logged "${LOG_DIR}/stage0_sft_smoke32.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT_BASE}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${SEMANTIC_DATA_DIR}" \
    --output-dir "${SMOKE_DIR}" \
    --max-length "${SFT_MAX_LENGTH}" \
    --limit-train 32 \
    --limit-val 32 \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --lr "${STAGE_A_LR}" \
    --lr-scheduler cosine \
    --warmup-steps 2 \
    --min-lr-ratio 0.2 \
    --atom-count-loss-weight "${ATOM_COUNT_LOSS_WEIGHT}" \
    --slot-marker-loss-weight "${SLOT_MARKER_LOSS_WEIGHT}" \
    --empty-slot-loss-weight "${EMPTY_SLOT_LOSS_WEIGHT}" \
    --nonempty-slot-loss-weight "${NONEMPTY_SLOT_LOSS_WEIGHT}" \
    --late-nonempty-slot-loss-weight "${LATE_NONEMPTY_SLOT_LOSS_WEIGHT}" \
    --coordinate-loss-weight "${COORDINATE_LOSS_WEIGHT}" \
    --pad-coordinate-loss-weight "${PAD_COORDINATE_LOSS_WEIGHT}" \
    --semantic-init-element-tokens \
    --element-token-alignment-loss-weight "${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT}" \
    --element-token-alignment-output-weight "${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT}" \
    --train-prefill-slot-tokens \
    --logging-steps 1 \
    --eval-steps 8 \
    --save-steps 1000 \
    --position-diagnostics-steps 8

train_one_epoch() {
  local label="$1"
  local output_dir="$2"
  local lr="$3"
  local warmup_steps="$4"
  local checkpoint_path="${5:-}"
  next_port
  local cmd=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/llada_sft.py
      --model-path "${MODEL_PATH}"
      --data-dir "${SEMANTIC_DATA_DIR}"
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
  if [ "${TRAIN_WEIGHTED_SAMPLING}" = "1" ]; then
    cmd+=(--weighted-sampling --weighted-sampling-power "${WEIGHTED_SAMPLING_POWER}")
    if [ -n "${SAMPLE_WEIGHT_MULTIPLIERS}" ]; then
      cmd+=(--sample-weight-multipliers "${SAMPLE_WEIGHT_MULTIPLIERS}")
    fi
  fi
  if [ -n "${checkpoint_path}" ]; then
    cmd+=(--checkpoint-path "${checkpoint_path}")
  fi
  run_logged "${LOG_DIR}/${label}_train.log" "${cmd[@]}"
}

summarize_training() {
  local label="$1"
  local output_dir="$2"
  local candidate_notes="${NOTES_DIR}/${label}"
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
trains = [e for e in events if e.get("event") == "train"]
evals = [e for e in events if e.get("event") == "eval"]
diags = [e for e in events if e.get("event") == "position_diagnostics"]
starts = [e for e in events if e.get("event") == "start"]
payload = {
    "label": "${label}",
    "training_log": str(log_path),
    "latest_train": trains[-1] if trains else {},
    "latest_eval": evals[-1] if evals else {},
    "latest_group_ce_summary": (diags[-1].get("group_ce_summary", {}) if diags else {}),
    "element_token_alignment": (starts[0].get("element_token_alignment", {}) if starts else {}),
}
Path("${candidate_notes}/training_diagnostics_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

smoke256() {
  local label="$1"
  local checkpoint="$2"
  local temperature="$3"
  local suffix="${4:-}"
  local sample_dir="${OUT_DIR}/${label}${suffix}_sample256"
  local candidate_notes="${NOTES_DIR}/${label}${suffix}"
  mkdir -p "${sample_dir}" "${candidate_notes}"
  next_port
  run_logged "${LOG_DIR}/${label}${suffix}_sample256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${sample_dir}" \
      --num-samples 256 \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --block-length 1 \
      --temperature "${temperature}" \
      --generation-schedule "${GENERATION_SCHEDULE}" \
      --schema-logit-mask \
      --prefill-slot-tokens \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask
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
    --min-parse-rate "${MIN_PARSE_RATE_256}" \
    --min-graph-acceptance 0.95 \
    --min-comp-valid "${MIN_COMP_VALID_256}" \
    --min-strict-valid "${MIN_STRICT_VALID_256}" \
    --max-single-element "${MAX_SINGLE_ELEMENT_256}" \
    --max-pbc-duplicate "${MAX_PBC_DUPLICATE_256}" \
    --output-json "${candidate_notes}/sample256_gate.json"
  python - <<PY
import json
from pathlib import Path
record = {"label": "${label}", "suffix": "${suffix}", "checkpoint": "${checkpoint}", "temperature": float("${temperature}")}
Path("${NOTES_DIR}/candidate_checkpoints.jsonl").open("a", encoding="utf-8").write(json.dumps(record, sort_keys=True) + "\n")
PY
}

: > "${NOTES_DIR}/candidate_checkpoints.jsonl"
previous_checkpoint=""
for epoch in $(seq 1 "${STAGE_A_EPOCHS}"); do
  label="stage_a_e${epoch}"
  output_dir="${OUT_DIR}/${label}"
  train_one_epoch "${label}" "${output_dir}" "${STAGE_A_LR}" "${STAGE_A_WARMUP_STEPS}" "${previous_checkpoint}"
  summarize_training "${label}" "${output_dir}" | tee "${LOG_DIR}/${label}_training_summary.log"
  smoke256 "${label}" "${output_dir}/final" "${TEMPERATURE}"
  previous_checkpoint="${output_dir}/final"
done

BEST_STAGE_A="$(python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
candidates = []
for line in (notes / "candidate_checkpoints.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if not rec["label"].startswith("stage_a_") or rec.get("suffix"):
        continue
    gate = json.loads((notes / rec["label"] / "sample256_gate.json").read_text())
    metrics = gate.get("metrics", {})
    candidates.append({
        **rec,
        "passed": bool(gate.get("passed")),
        "parse_rate": float(metrics.get("parse_rate") or 0),
        "graph_acceptance": float(metrics.get("graph_acceptance") or 0),
        "comp_valid": float(metrics.get("comp_valid") or 0),
        "strict_valid": float(metrics.get("strict_valid") or 0),
        "single_element": float(metrics.get("single_element") or 0),
    })
candidates.sort(key=lambda x: (x["passed"], x["parse_rate"] >= 0.98, x["graph_acceptance"], x["comp_valid"], x["strict_valid"], -x["single_element"]), reverse=True)
payload = {"stage": "stage_a", "candidates": candidates, "best": candidates[0] if candidates else None}
(notes / "stage_a_candidate_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(payload["best"]["checkpoint"] if payload["best"] else "")
PY
)"
STAGE_A_PASSED="$(python - <<PY
import json
from pathlib import Path
p=json.loads(Path("${NOTES_DIR}/stage_a_candidate_selection.json").read_text()).get("best") or {}
print("1" if p.get("passed") else "0")
PY
)"

if [ "${STAGE_A_PASSED}" != "1" ] && [ "${FORCE_STAGE_B}" != "1" ]; then
  echo "Best Stage A failed 256 gate; stopping before Stage B/1000."
  cleanup_weight_dirs_except "${BEST_STAGE_A}" "${previous_checkpoint}"
  exit 0
fi

previous_checkpoint="${BEST_STAGE_A}"
for epoch in $(seq 1 "${STAGE_B_EPOCHS}"); do
  label="stage_b_e${epoch}"
  output_dir="${OUT_DIR}/${label}"
  train_one_epoch "${label}" "${output_dir}" "${STAGE_B_LR}" "${STAGE_B_WARMUP_STEPS}" "${previous_checkpoint}"
  summarize_training "${label}" "${output_dir}" | tee "${LOG_DIR}/${label}_training_summary.log"
  smoke256 "${label}" "${output_dir}/final" "${TEMPERATURE}"
  previous_checkpoint="${output_dir}/final"
done

BEST_CHECKPOINT="$(python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
candidates = []
for line in (notes / "candidate_checkpoints.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if rec.get("suffix"):
        continue
    gate = json.loads((notes / rec["label"] / "sample256_gate.json").read_text())
    metrics = gate.get("metrics", {})
    failure_modes_path = notes / rec["label"] / "sample256_failure_modes.json"
    failure_modes = json.loads(failure_modes_path.read_text()) if failure_modes_path.exists() else {}
    reason_counts = failure_modes.get("reason_counts", {})
    candidates.append({
        **rec,
        "passed": bool(gate.get("passed")),
        "parse_rate": float(metrics.get("parse_rate") or 0),
        "graph_acceptance": float(metrics.get("graph_acceptance") or 0),
        "comp_valid": float(metrics.get("comp_valid") or 0),
        "strict_valid": float(metrics.get("strict_valid") or 0),
        "single_element": float(metrics.get("single_element") or 0),
        "all_metal": float(metrics.get("all_metal") or 0),
        "pbc_duplicate": float(metrics.get("pbc_duplicate") or 0),
        "charge_neutrality_fail_count": int(reason_counts.get("charge_neutrality_fail", 0)),
    })
candidates.sort(
    key=lambda x: (
        x["passed"],
        x["parse_rate"] >= 0.98,
        x["graph_acceptance"],
        x["comp_valid"],
        x["strict_valid"],
        -x["single_element"],
        -x["pbc_duplicate"],
        -x["charge_neutrality_fail_count"],
    ),
    reverse=True,
)
payload = {"candidates": candidates, "best": candidates[0] if candidates else None}
(notes / "candidate_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(payload["best"]["checkpoint"] if payload["best"] else "")
PY
)"
BEST_LABEL="$(python - <<PY
import json
from pathlib import Path
p=json.loads(Path("${NOTES_DIR}/candidate_selection.json").read_text()).get("best") or {}
print(p.get("label",""))
PY
)"
LATEST_CHECKPOINT="${previous_checkpoint}"
echo "BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "LATEST_CHECKPOINT=${LATEST_CHECKPOINT}"
cleanup_weight_dirs_except "${BEST_CHECKPOINT}" "${LATEST_CHECKPOINT}"

BEST_PASSED="$(python - <<PY
import json
from pathlib import Path
p=json.loads(Path("${NOTES_DIR}/candidate_selection.json").read_text()).get("best") or {}
print("1" if p.get("passed") else "0")
PY
)"
if [ "${BEST_PASSED}" != "1" ] && [ "${FORCE_1000}" != "1" ]; then
  echo "No candidate passed 256 gate; stopping before 1000/refinement/SUN."
  exit 0
fi

if [ "${RUN_TEMPERATURE_COMPARE}" = "1" ]; then
  smoke256 "${BEST_LABEL}" "${BEST_CHECKPOINT}" "${TEMPERATURE_COMPARE}" "_temp${TEMPERATURE_COMPARE/./}"
fi

SAMPLE1000_DIR="${OUT_DIR}/sample1000"
REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/mattergen_sun1000"
mkdir -p "${SAMPLE1000_DIR}" "${REFINED1000_DIR}" "${SUN1000_DIR}"

next_port
run_logged "${LOG_DIR}/sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --output-dir "${SAMPLE1000_DIR}" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --num-samples "${MAX_ATTEMPTS}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --block-length 1 \
    --temperature "${TEMPERATURE}" \
    --generation-schedule "${GENERATION_SCHEDULE}" \
    --schema-logit-mask \
    --prefill-slot-tokens \
    --atom-count-grammar-mask \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

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
  --relax-max-steps "${RELAX_MAX_STEPS}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH}"
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

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
def read(name):
    p = notes / name
    return json.loads(p.read_text()) if p.exists() else {}
payload = {
    "run_id": "${RUN_ID}",
    "best_label": "${BEST_LABEL}",
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "latest_checkpoint": "${LATEST_CHECKPOINT}",
    "candidate_selection": read("candidate_selection.json"),
    "sample1000": json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()),
    "composition1000": read("composition1000.json"),
    "crysllmgen": read("crysllmgen_metrics1000.json"),
    "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
    "no_10000_evaluation": True,
}
(notes / "result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
report = Path("reports") / f"{'${RUN_ID}'}_formula_semantic_restart_report.md"
lines = [
    f"# ${RUN_ID} Formula-Semantic Fixed-Slot Restart",
    "",
    f"- best checkpoint: `{payload['best_checkpoint']}`",
    f"- latest checkpoint: `{payload['latest_checkpoint']}`",
    f"- temperature: `${TEMPERATURE}`",
    f"- no 10000 evaluation: `true`",
    "",
    "## Candidate Selection",
    "",
    "```json",
    json.dumps(payload["candidate_selection"], indent=2, sort_keys=True),
    "```",
    "",
    "## 1000 Evaluation",
    "",
    "```json",
    json.dumps({k: payload[k] for k in ("sample1000", "crysllmgen", "sun_thresholds")}, indent=2, sort_keys=True),
    "```",
]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

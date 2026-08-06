#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260526_stalign_restart_r1}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
STAGE_A_CHECKPOINT_PATH="${STAGE_A_CHECKPOINT_PATH:-}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
WEIGHTED_DATA_DIR="${WEIGHTED_DATA_DIR:-data/dlm_sft/mp_20_ehull_weighted_r0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
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
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT:-2}}"
FORCE_STAGE_B="${FORCE_STAGE_B:-0}"
FORCE_1000="${FORCE_1000:-0}"
MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.85}"
MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.40}"
MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
MAX_PBC_DUPLICATE_256="${MAX_PBC_DUPLICATE_256:-0.0}"
STAGE_A_LR="${STAGE_A_LR:-1e-4}"
STAGE_A_WARMUP_STEPS="${STAGE_A_WARMUP_STEPS:-50}"
STAGE_B_LR="${STAGE_B_LR:-2e-5}"
STAGE_B_WARMUP_STEPS="${STAGE_B_WARMUP_STEPS:-50}"
WEIGHTED_SAMPLING="${WEIGHTED_SAMPLING:-1}"
WEIGHTED_SAMPLING_POWER="${WEIGHTED_SAMPLING_POWER:-1.0}"
SAMPLE_WEIGHT_MULTIPLIERS="${SAMPLE_WEIGHT_MULTIPLIERS:-}"
IGNORE_JSONL_SAMPLE_WEIGHT="${IGNORE_JSONL_SAMPLE_WEIGHT:-0}"
ATOM_COUNT_LOSS_WEIGHT="${ATOM_COUNT_LOSS_WEIGHT:-3.0}"
SLOT_MARKER_LOSS_WEIGHT="${SLOT_MARKER_LOSS_WEIGHT:-0.25}"
EMPTY_SLOT_LOSS_WEIGHT="${EMPTY_SLOT_LOSS_WEIGHT:-0.5}"
NONEMPTY_SLOT_LOSS_WEIGHT="${NONEMPTY_SLOT_LOSS_WEIGHT:-2.0}"
LATE_NONEMPTY_SLOT_LOSS_WEIGHT="${LATE_NONEMPTY_SLOT_LOSS_WEIGHT:-4.0}"
COORDINATE_LOSS_WEIGHT="${COORDINATE_LOSS_WEIGHT:-1.0}"
PAD_COORDINATE_LOSS_WEIGHT="${PAD_COORDINATE_LOSS_WEIGHT:-0.2}"
ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT:-0.0}"
ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT:-1.0}"
SUN_BUFFER_ACCEPTED_TIERS="${SUN_BUFFER_ACCEPTED_TIERS:-strict,meta}"
SUN_BUFFER_ACCEPTED_REASONS="${SUN_BUFFER_ACCEPTED_REASONS:-charge_neutral_pauling_valid,all_metal_shortcut}"
SUN_BUFFER_MAX_FORMULA_REPEATS="${SUN_BUFFER_MAX_FORMULA_REPEATS:-4}"
SUN_BUFFER_MAX_CHEMSYS_REPEATS="${SUN_BUFFER_MAX_CHEMSYS_REPEATS:-32}"
MASTER_PORT_BASE="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"

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

cleanup_weight_files_except() {
  local keep_a="${1:-}"
  local keep_b="${2:-}"
  python - <<PY
from pathlib import Path

roots = [Path("${OUT_DIR}")]
keep = {Path(p).resolve() for p in ["${keep_a}", "${keep_b}"] if p}
patterns = ("*.safetensors", "*.bin", "*.pt")
deleted = []
for root in roots:
    if not root.exists():
        continue
    for pattern in patterns:
        for path in root.rglob(pattern):
            try:
                resolved = path.resolve()
                if any(resolved.is_relative_to(k) for k in keep):
                    continue
            except AttributeError:
                if any(str(resolved).startswith(str(k) + "/") or resolved == k for k in keep):
                    continue
            deleted.append(str(path))
            path.unlink()
Path("${NOTES_DIR}/weight_cleanup_last.json").write_text(
    __import__("json").dumps({"kept": [str(k) for k in keep], "deleted": deleted}, indent=2, sort_keys=True) + "\n"
)
print(f"deleted_weight_files={len(deleted)}")
PY
}

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "model_path": "${MODEL_PATH}",
  "stage_a_checkpoint_path": "${STAGE_A_CHECKPOINT_PATH}",
  "weighted_data_dir": "${WEIGHTED_DATA_DIR}",
  "temperature": float("${TEMPERATURE}"),
  "generation_schedule": "${GENERATION_SCHEDULE}",
  "target_graph_success": int("${TARGET_GRAPH_SUCCESS}"),
  "max_attempts": int("${MAX_ATTEMPTS}"),
  "nproc_per_node": int("${NPROC_PER_NODE}"),
  "sft_batch_size": int("${SFT_BATCH_SIZE}"),
  "sft_grad_accum": int("${SFT_GRAD_ACCUM}"),
  "force_stage_b": bool(int("${FORCE_STAGE_B}")),
  "force_1000": bool(int("${FORCE_1000}")),
  "smoke256_thresholds": {
    "min_comp_valid": float("${MIN_COMP_VALID_256}"),
    "min_strict_valid": float("${MIN_STRICT_VALID_256}"),
    "max_single_element": float("${MAX_SINGLE_ELEMENT_256}"),
    "max_pbc_duplicate": float("${MAX_PBC_DUPLICATE_256}")
  },
  "stage_a_lr": "${STAGE_A_LR}",
  "stage_a_warmup_steps": int("${STAGE_A_WARMUP_STEPS}"),
  "stage_b_lr": "${STAGE_B_LR}",
  "stage_b_warmup_steps": int("${STAGE_B_WARMUP_STEPS}"),
  "weighted_sampling": bool(int("${WEIGHTED_SAMPLING}")),
  "weighted_sampling_power": float("${WEIGHTED_SAMPLING_POWER}"),
  "sample_weight_multipliers": "${SAMPLE_WEIGHT_MULTIPLIERS}",
  "ignore_jsonl_sample_weight": bool(int("${IGNORE_JSONL_SAMPLE_WEIGHT}")),
  "element_token_alignment_loss_weight": float("${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT}"),
  "element_token_alignment_output_weight": float("${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT}"),
  "loss_weights": {
    "atom_count": float("${ATOM_COUNT_LOSS_WEIGHT}"),
    "slot_marker": float("${SLOT_MARKER_LOSS_WEIGHT}"),
    "empty_slot": float("${EMPTY_SLOT_LOSS_WEIGHT}"),
    "nonempty_slot": float("${NONEMPTY_SLOT_LOSS_WEIGHT}"),
    "late_nonempty_slot": float("${LATE_NONEMPTY_SLOT_LOSS_WEIGHT}"),
    "coordinate": float("${COORDINATE_LOSS_WEIGHT}"),
    "pad_coordinate": float("${PAD_COORDINATE_LOSS_WEIGHT}")
  },
  "no_10000_evaluation": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [ ! -f "${WEIGHTED_DATA_DIR}/train.jsonl" ]; then
  if [ ! -f "${BASE_DATA_DIR}/train.jsonl" ]; then
    run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
      python scripts/build_crystal_sft_data.py \
        --input-dir reference/crysllmgen/data/mp_20 \
        --output-dir "${BASE_DATA_DIR}" \
        --tokenizer-path "${MODEL_PATH}" \
        --answer-separator ""
  fi
  run_logged "${LOG_DIR}/build_ehull_weighted_data.log" \
    python scripts/build_mp20_ehull_weighted_sft_data.py \
      --base-dir "${BASE_DATA_DIR}" \
      --csv-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${WEIGHTED_DATA_DIR}" \
      --extra-fraction 0.15 \
      --max-formula-repeats 8 \
      --max-chemsys-repeats 64
fi

train_stage() {
  local stage_name="$1"
  local output_dir="$2"
  local lr="$3"
  local warmup_steps="$4"
  local checkpoint_path="${5:-}"
  next_port
  local cmd=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/llada_sft.py
      --model-path "${MODEL_PATH}"
  )
  if [ -n "${checkpoint_path}" ]; then
    cmd+=(--checkpoint-path "${checkpoint_path}")
  fi
  cmd+=(
      --data-dir "${WEIGHTED_DATA_DIR}"
      --output-dir "${output_dir}"
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

smoke256() {
  local candidate="$1"
  local checkpoint="$2"
  local sample_dir="${OUT_DIR}/${candidate}_sample256"
  local candidate_notes="${NOTES_DIR}/${candidate}"
  mkdir -p "${sample_dir}" "${candidate_notes}"
  next_port
  run_logged "${LOG_DIR}/${candidate}_sample256.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${sample_dir}" \
      --num-samples 256 \
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

STAGE_A_DIR="${OUT_DIR}/stage_a"
train_stage "stage_a" "${STAGE_A_DIR}" "${STAGE_A_LR}" "${STAGE_A_WARMUP_STEPS}" "${STAGE_A_CHECKPOINT_PATH}"
summarize_training_log "stage_a" "${STAGE_A_DIR}" | tee "${LOG_DIR}/stage_a_training_diagnostics_summary.log"
smoke256 "stage_a" "${STAGE_A_DIR}/final"

STAGE_A_PASS="$(python - <<PY
import json
from pathlib import Path
gate=json.loads(Path("${NOTES_DIR}/stage_a/sample256_gate.json").read_text())
print("1" if gate.get("passed") else "0")
PY
)"
if [ "${STAGE_A_PASS}" != "1" ] && [ "${FORCE_STAGE_B}" != "1" ]; then
  echo "Stage A failed 256 gate; stopping before Stage B/1000."
  cleanup_weight_files_except
  exit 0
elif [ "${STAGE_A_PASS}" != "1" ]; then
  echo "Stage A failed strict 256 gate, but FORCE_STAGE_B=1; continuing to Stage B for the conservative second epoch."
fi

STAGE_B_DIR="${OUT_DIR}/stage_b"
train_stage "stage_b" "${STAGE_B_DIR}" "${STAGE_B_LR}" "${STAGE_B_WARMUP_STEPS}" "${STAGE_A_DIR}/final"
summarize_training_log "stage_b" "${STAGE_B_DIR}" | tee "${LOG_DIR}/stage_b_training_diagnostics_summary.log"
smoke256 "stage_b" "${STAGE_B_DIR}/final"

BEST_CHECKPOINT="$(python - <<PY
import json
from pathlib import Path
candidates=[]
for name,path in [("stage_a","${STAGE_A_DIR}/final"),("stage_b","${STAGE_B_DIR}/final")]:
    gate=json.loads(Path("${NOTES_DIR}") .joinpath(name,"sample256_gate.json").read_text())
    metrics=gate.get("metrics",{})
    reasons=(json.loads(Path("${NOTES_DIR}") .joinpath(name,"sample256_failure_modes.json").read_text()).get("reason_counts",{}))
    train_diag_path = Path("${NOTES_DIR}") .joinpath(name,"training_diagnostics_summary.json")
    train_diag = json.loads(train_diag_path.read_text()) if train_diag_path.exists() else {}
    candidates.append({
        "name": name,
        "checkpoint": path,
        "passed": gate.get("passed", False),
        "comp_valid": float(metrics.get("comp_valid") or 0),
        "strict_valid": float(metrics.get("strict_valid") or 0),
        "charge_fail": int(reasons.get("charge_neutrality_fail", 999999)),
        "latest_val_loss": train_diag.get("latest_eval", {}).get("val_loss"),
        "slot_marker_free_ce": train_diag.get("latest_group_ce_summary", {}).get("slot_marker_free_ce"),
        "group_ce_summary": train_diag.get("latest_group_ce_summary", {}),
        "weighted_sampling": train_diag.get("weighted_sampling", {}),
    })
candidates.sort(
    key=lambda x: (
        x["passed"],
        x["comp_valid"],
        x["strict_valid"],
        -x["charge_fail"],
        -999.0 if x.get("slot_marker_free_ce") is None else -float(x["slot_marker_free_ce"]),
    ),
    reverse=True,
)
Path("${NOTES_DIR}/candidate_selection.json").write_text(json.dumps({"candidates": candidates, "best": candidates[0]}, indent=2, sort_keys=True)+"\n")
print(candidates[0]["checkpoint"])
PY
)"
export BEST_CHECKPOINT
echo "BEST_CHECKPOINT=${BEST_CHECKPOINT}"
LATEST_CHECKPOINT="${STAGE_B_DIR}/final"
cleanup_weight_files_except "${BEST_CHECKPOINT}" "${LATEST_CHECKPOINT}"
BEST_PASSED="$(python - <<PY
import json
from pathlib import Path
payload=json.loads(Path("${NOTES_DIR}/candidate_selection.json").read_text())
print("1" if payload.get("best", {}).get("passed") else "0")
PY
)"
if [ "${BEST_PASSED}" != "1" ] && [ "${FORCE_1000}" != "1" ]; then
  echo "No candidate passed 256 gate; stopping before 1000/refinement/SUN."
  cleanup_weight_files_except
  exit 0
elif [ "${BEST_PASSED}" != "1" ]; then
  echo "No candidate passed 256 gate, but FORCE_1000=1; continuing for diagnostic 1000."
fi

SAMPLE1000_DIR="${OUT_DIR}/sample1000"
REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/mattergen_sun1000"
BUFFER_DIR="data/dlm_sft/mp_20_strict_sun_buffer_r1"
mkdir -p "${SAMPLE1000_DIR}" "${REFINED1000_DIR}" "${SUN1000_DIR}" "${BUFFER_DIR}"

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
  --accepted-tiers "${SUN_BUFFER_ACCEPTED_TIERS}" \
  --accepted-composition-reasons "${SUN_BUFFER_ACCEPTED_REASONS}" \
  --max-formula-repeats "${SUN_BUFFER_MAX_FORMULA_REPEATS}" \
  --max-chemsys-repeats "${SUN_BUFFER_MAX_CHEMSYS_REPEATS}"
cp "${BUFFER_DIR}/strict_sun_success_summary.json" "${NOTES_DIR}/strict_sun_success_summary.json"

python - <<PY
import json
from pathlib import Path
notes=Path("${NOTES_DIR}")
def read(name):
    p=notes/name
    return json.loads(p.read_text()) if p.exists() else {}
payload={
  "run_id": "${RUN_ID}",
  "best_checkpoint": "${BEST_CHECKPOINT}",
  "candidate_selection": read("candidate_selection.json"),
  "sample1000": json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()),
  "crysllmgen": read("crysllmgen_metrics1000.json"),
  "composition1000": read("composition1000.json"),
  "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
  "strict_sun_buffer": read("strict_sun_success_summary.json"),
  "no_10000_evaluation": True,
}
(notes/"result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
PY

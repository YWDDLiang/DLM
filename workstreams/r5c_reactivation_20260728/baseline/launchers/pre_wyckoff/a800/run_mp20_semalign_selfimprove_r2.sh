#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260527_semalign_selfimprove_r2}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PREV_RUN_ID="${PREV_RUN_ID:-20260527_compfirst_semalign_lowlr5_r1}"
PREV_RUN_DIR="${PREV_RUN_DIR:-runs/${PREV_RUN_ID}}"
PREV_BEST_CHECKPOINT="${PREV_BEST_CHECKPOINT:-${PREV_RUN_DIR}/outputs/stage_b/final}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
BUFFER_DIR="${BUFFER_DIR:-data/dlm_sft/mp_20_sun_self_improve_${RUN_ID}}"
WEIGHTED_DATA_DIR="${WEIGHTED_DATA_DIR:-data/dlm_sft/mp_20_sun_self_improve_weighted_${RUN_ID}}"
BUFFER_JSONL="${BUFFER_JSONL:-${BUFFER_DIR}/sun_self_improving_success.jsonl}"
NOTES_DIR="runs/${RUN_ID}/notes"
LOG_DIR="runs/${RUN_ID}/logs"

cd "${PROJECT_ROOT}"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${BUFFER_DIR}"

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

if [ ! -f "${BASE_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

if [ "${REBUILD_SELF_IMPROVING_BUFFER:-0}" = "1" ]; then
  rm -f "${BUFFER_JSONL}" "${BUFFER_DIR}/sun_self_improving_success_summary.json"
fi

if [ ! -f "${BUFFER_JSONL}" ]; then
  run_logged "${LOG_DIR}/build_sun_self_improving_buffer.log" \
    python scripts/build_strict_sun_self_improving_buffer.py \
      --relaxed-extxyz "${PREV_RUN_DIR}/outputs/mattergen_sun1000/relaxed.extxyz" \
      --mattergen-summary-json "${PREV_RUN_DIR}/notes/mattergen_sun1000_summary.json" \
      --mattergen-detailed-json "${PREV_RUN_DIR}/notes/mattergen_sun1000_detailed_metrics.json" \
      --refined-pt "${PREV_RUN_DIR}/outputs/refined1000/dlm_refined_mp_1000.pt" \
      --raw-generations-jsonl "${PREV_RUN_DIR}/outputs/sample1000/raw_generations.jsonl" \
      --output-jsonl "${BUFFER_JSONL}" \
      --summary-json "${BUFFER_DIR}/sun_self_improving_success_summary.json" \
      --accepted-tiers "${SELF_IMPROVING_ACCEPTED_TIERS:-strict,meta}" \
      --accepted-composition-reasons "${SELF_IMPROVING_ACCEPTED_REASONS:-charge_neutral_pauling_valid,all_metal_shortcut}" \
      --max-formula-repeats "${SELF_IMPROVING_MAX_FORMULA_REPEATS:-4}" \
      --max-chemsys-repeats "${SELF_IMPROVING_MAX_CHEMSYS_REPEATS:-32}"
fi
cp "${BUFFER_DIR}/sun_self_improving_success_summary.json" "${NOTES_DIR}/input_sun_self_improving_success_summary.json"

if [ "${REBUILD_WEIGHTED_SELF_IMPROVING_DATA:-0}" = "1" ]; then
  rm -rf "${WEIGHTED_DATA_DIR}"
fi

if [ ! -f "${WEIGHTED_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_self_improving_weighted_data.log" \
    python scripts/build_mp20_ehull_weighted_sft_data.py \
      --base-dir "${BASE_DATA_DIR}" \
      --csv-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${WEIGHTED_DATA_DIR}" \
      --self-improving-jsonl "${BUFFER_JSONL}" \
      --include-meta-self-improving \
      --self-improving-repeat-with-replacement \
      --extra-fraction "${EXTRA_FRACTION:-0.10}" \
      --self-improving-fraction "${SELF_IMPROVING_FRACTION:-0.06}" \
      --max-formula-repeats "${WEIGHTED_MAX_FORMULA_REPEATS:-4}" \
      --max-chemsys-repeats "${WEIGHTED_MAX_CHEMSYS_REPEATS:-32}" \
      --tier-weights "${TIER_WEIGHTS:-tier_high=1.0,tier_mid_high=0.75,tier_mid=0.45,tier_low_mid=0.20,tier_low_retained=0.08}" \
      --mask-policy-mix "${MASK_POLICY_MIX:-normal=0.45,active_element=0.15,n_active_element=0.25,active_element_empty=0.15}"
fi
cp "${WEIGHTED_DATA_DIR}/ehull_weight_summary.json" "${NOTES_DIR}/input_weighted_data_summary.json"

export RUN_ID
export PROJECT_ROOT
export MODEL_PATH
export STAGE_A_CHECKPOINT_PATH="${STAGE_A_CHECKPOINT_PATH:-${PREV_BEST_CHECKPOINT}}"
export WEIGHTED_DATA_DIR
export WEIGHTED_SAMPLING="${WEIGHTED_SAMPLING:-1}"
export IGNORE_JSONL_SAMPLE_WEIGHT="${IGNORE_JSONL_SAMPLE_WEIGHT:-0}"
export SAMPLE_WEIGHT_MULTIPLIERS="${SAMPLE_WEIGHT_MULTIPLIERS:-all_metal=0.35,single_element=0.02,invalid=0.15,selection_role:self_improving_repeat=1.3}"
export STAGE_A_LR="${STAGE_A_LR:-2e-6}"
export STAGE_A_WARMUP_STEPS="${STAGE_A_WARMUP_STEPS:-20}"
export STAGE_B_LR="${STAGE_B_LR:-8e-7}"
export STAGE_B_WARMUP_STEPS="${STAGE_B_WARMUP_STEPS:-20}"
export ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_LOSS_WEIGHT:-0.08}"
export ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT="${ELEMENT_TOKEN_ALIGNMENT_OUTPUT_WEIGHT:-1.0}"
export ATOM_COUNT_LOSS_WEIGHT="${ATOM_COUNT_LOSS_WEIGHT:-3.0}"
export EMPTY_SLOT_LOSS_WEIGHT="${EMPTY_SLOT_LOSS_WEIGHT:-0.20}"
export NONEMPTY_SLOT_LOSS_WEIGHT="${NONEMPTY_SLOT_LOSS_WEIGHT:-2.5}"
export LATE_NONEMPTY_SLOT_LOSS_WEIGHT="${LATE_NONEMPTY_SLOT_LOSS_WEIGHT:-4.0}"
export COORDINATE_LOSS_WEIGHT="${COORDINATE_LOSS_WEIGHT:-1.2}"
export PAD_COORDINATE_LOSS_WEIGHT="${PAD_COORDINATE_LOSS_WEIGHT:-0.10}"
export MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.88}"
export MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.40}"
export MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
export SUN_BUFFER_ACCEPTED_TIERS="${SUN_BUFFER_ACCEPTED_TIERS:-strict,meta}"
export SUN_BUFFER_ACCEPTED_REASONS="${SUN_BUFFER_ACCEPTED_REASONS:-charge_neutral_pauling_valid,all_metal_shortcut}"
export SUN_BUFFER_MAX_FORMULA_REPEATS="${SUN_BUFFER_MAX_FORMULA_REPEATS:-4}"
export SUN_BUFFER_MAX_CHEMSYS_REPEATS="${SUN_BUFFER_MAX_CHEMSYS_REPEATS:-32}"

bash scripts/a800/run_mp20_stalign_restart_r1.sh

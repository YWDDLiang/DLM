#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
START_CHECKPOINT="${START_CHECKPOINT:-${1:-runs/20260521_211500-final07-refined-seal1/outputs/sft_refined_seal1/final}}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20_reason_balanced_v3_preserve_select}"
REFINED_BUFFER_JSONL="${REFINED_BUFFER_JSONL:-data/dlm_sft/mp_20_refined_seal_r1/seal_success.jsonl}"
REFINED_PT="${REFINED_PT:-}"
RAW_GENERATIONS_JSONL="${RAW_GENERATIONS_JSONL:-}"
REFINED_BUFFER_DIR="${REFINED_BUFFER_DIR:-data/dlm_sft/mp_20_refined_seal_auto}"
AUG_DATA_DIR="${AUG_DATA_DIR:-data/dlm_sft/mp_20_refined_seal_aug15}"

EXTRA_FRACTION="${EXTRA_FRACTION:-0.15}"
LR="${LR:-3e-7}"
TEMPERATURE="${TEMPERATURE:-0.7}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-default}"
AUGMENT_EXTRA_ORIGIN_SHIFT="${AUGMENT_EXTRA_ORIGIN_SHIFT:-0}"
AUGMENT_EXTRA_SITE_PERMUTATION="${AUGMENT_EXTRA_SITE_PERMUTATION:-0}"

RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/outputs/sft_refined_aug15" "${RUN_DIR}/outputs/sample256" "${RUN_DIR}/notes"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "start_checkpoint": "${START_CHECKPOINT}",
    "base_data_dir": "${BASE_DATA_DIR}",
    "refined_buffer_jsonl": "${REFINED_BUFFER_JSONL}",
    "aug_data_dir": "${AUG_DATA_DIR}",
    "extra_fraction": float("${EXTRA_FRACTION}"),
    "lr": float("${LR}"),
    "temperature": float("${TEMPERATURE}"),
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "epochs": int("${EPOCHS}"),
    "batch_size": int("${BATCH_SIZE}"),
    "grad_accum": int("${GRAD_ACCUM}"),
    "augment_extra_origin_shift": bool(int("${AUGMENT_EXTRA_ORIGIN_SHIFT}")),
    "augment_extra_site_permutation": bool(int("${AUGMENT_EXTRA_SITE_PERMUTATION}")),
    "sft_mode": "refined_only_seal_mask_policy_augmented",
}, indent=2) + "\n")
PY

if [ -n "${REFINED_PT}" ]; then
  mkdir -p "${REFINED_BUFFER_DIR}"
  buffer_args=(
    --refined-pt "${REFINED_PT}"
    --output-jsonl "${REFINED_BUFFER_DIR}/seal_success.jsonl"
    --summary-json "${REFINED_BUFFER_DIR}/seal_success_summary.json"
    --refined-world-size 2
    --max-formula-repeats 8
    --max-single-fraction 0.10
    --max-all-metal-fraction 0.60
  )
  if [ -n "${RAW_GENERATIONS_JSONL}" ]; then
    buffer_args+=(--raw-generations-jsonl "${RAW_GENERATIONS_JSONL}")
  fi
  python scripts/build_refined_seal_success_buffer.py "${buffer_args[@]}"
  REFINED_BUFFER_JSONL="${REFINED_BUFFER_DIR}/seal_success.jsonl"
  cp "${REFINED_BUFFER_DIR}/seal_success_summary.json" "${RUN_DIR}/notes/seal_success_summary.json"
fi

build_args=(
  --base-dir "${BASE_DATA_DIR}"
  --output-dir "${AUG_DATA_DIR}"
  --extra-jsonl "${REFINED_BUFFER_JSONL}"
  --extra-fraction "${EXTRA_FRACTION}"
  --accepted-buckets strict,all_metal,single_element
  --max-formula-repeats 8
  --repeat-extra-to-target
  --augmentation-mask-policies n_active_element,active_element,active_element_empty,normal
)
if [ "${AUGMENT_EXTRA_ORIGIN_SHIFT}" = "1" ]; then
  build_args+=(--augment-extra-origin-shift)
fi
if [ "${AUGMENT_EXTRA_SITE_PERMUTATION}" = "1" ]; then
  build_args+=(--augment-extra-site-permutation)
fi

python scripts/build_sft_data_with_extra_buffer.py "${build_args[@]}"

cp "${AUG_DATA_DIR}/seal_mix_summary.json" "${RUN_DIR}/notes/seal_mix_summary.json"

torchrun --nproc_per_node=2 scripts/llada_sft.py \
  --checkpoint-path "${START_CHECKPOINT}" \
  --data-dir "${AUG_DATA_DIR}" \
  --output-dir "${RUN_DIR}/outputs/sft_refined_aug15" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --lr-scheduler cosine \
  --warmup-steps 50 \
  --min-lr-ratio 0.2 \
  --atom-count-loss-weight 1.2 \
  --nonempty-slot-loss-weight 1.5 \
  --empty-slot-loss-weight 1.2 \
  --coordinate-loss-weight 1.0 \
  --pad-coordinate-loss-weight 0.2 \
  --position-diagnostics-steps 500 \
  --logging-steps 20 \
  --eval-steps 500 \
  --save-steps 1000

sample_args=(
  --checkpoint-path "${RUN_DIR}/outputs/sft_refined_aug15/final" \
  --output-dir "${RUN_DIR}/outputs/sample256" \
  --num-samples "${SAMPLE_COUNT}" \
  --batch-size "${SAMPLE_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}"
)
if [ "${GENERATION_SCHEDULE}" != "default" ]; then
  sample_args+=(--generation-schedule "${GENERATION_SCHEDULE}")
fi

torchrun --nproc_per_node=2 scripts/sample_llada_crystals.py "${sample_args[@]}"

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --failure-jsonl "${RUN_DIR}/outputs/sample256/failure_cases.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_distribution.json" \
  --output-md "${RUN_DIR}/notes/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_composition.json" \
  --output-md "${RUN_DIR}/notes/sample256_composition.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_failure_modes.json" \
  --output-md "${RUN_DIR}/notes/sample256_failure_modes.md"

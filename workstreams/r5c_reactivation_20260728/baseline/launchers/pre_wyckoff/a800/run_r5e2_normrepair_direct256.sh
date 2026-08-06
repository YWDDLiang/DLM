#!/usr/bin/env bash
set -Eeuo pipefail

# R5-E2 keeps generation de novo and unconditional:
# one direct compact-plan generation per sample, then at most one learned repair
# pass for flagged rows. No candidate pools, no K/N conditioning, no weighted
# sampling prior.

export START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-20260530_063846-r5e-trainrepair-direct256}"
export START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-runs/${START_PARENT_RUN_ID}/outputs/r5e_plan_compact_trainrepair_sft/final}"
export PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_compact_normrepair_v1}"
export WEIGHT_PROFILE="${WEIGHT_PROFILE:-default}"
export WEIGHTED_SAMPLING="${WEIGHTED_SAMPLING:-0}"
export REPAIR_AUGMENTATIONS_PER_ROW="${REPAIR_AUGMENTATIONS_PER_ROW:-4}"
export REPAIR_SAMPLE_WEIGHT="${REPAIR_SAMPLE_WEIGHT:-1.0}"
export TRAIN_LR="${TRAIN_LR:-1e-5}"
export TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-80}"
export TEMPERATURE="${TEMPERATURE:-0.60}"
export REPAIR_TEMPERATURE="${REPAIR_TEMPERATURE:-0.35}"

exec "$(dirname "$0")/run_r5e_trainrepair_direct256.sh"

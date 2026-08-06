#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLAN_FORMAT="${PLAN_FORMAT:-countfields}"
export PLAN_NAME="${PLAN_NAME:-r5d11_countfields_all}"
export PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_countfields_all_v2}"
export COMPOSITION_FILTER="${COMPOSITION_FILTER:-all}"
export SAMPLE_WEIGHT_PROFILE="${SAMPLE_WEIGHT_PROFILE:-uniform}"
export START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-base_llada}"
export START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-none}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"

exec "${SCRIPT_DIR}/run_r5d5_atomseq_direct256.sh"

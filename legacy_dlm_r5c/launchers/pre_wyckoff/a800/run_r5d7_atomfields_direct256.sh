#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLAN_FORMAT="${PLAN_FORMAT:-atomfields}"
export PLAN_NAME="${PLAN_NAME:-r5d7_atomfields}"
export PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_atomfields_smact_v1}"
export START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-20260530_1350-r5d6-atomslots-direct256}"
export START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-runs/${START_PARENT_RUN_ID}/outputs/r5d6_atomslots_plan_sft/final}"

exec "${SCRIPT_DIR}/run_r5d5_atomseq_direct256.sh"

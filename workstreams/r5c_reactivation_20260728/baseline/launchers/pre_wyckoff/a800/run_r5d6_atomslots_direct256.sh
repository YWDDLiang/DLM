#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLAN_FORMAT="${PLAN_FORMAT:-atomslots}"
export PLAN_NAME="${PLAN_NAME:-r5d6_atomslots}"
export PLAN_DATA_DIR="${PLAN_DATA_DIR:-data/dlm_sft/mp_20_r5_plan_atomslots_smact_v1}"
export START_PARENT_RUN_ID="${START_PARENT_RUN_ID:-20260530_1240-r5d5-atomseq-direct256}"
export START_CHECKPOINT_PATH="${START_CHECKPOINT_PATH:-runs/${START_PARENT_RUN_ID}/outputs/r5d5_atomseq_plan_sft/final}"

exec "${SCRIPT_DIR}/run_r5d5_atomseq_direct256.sh"

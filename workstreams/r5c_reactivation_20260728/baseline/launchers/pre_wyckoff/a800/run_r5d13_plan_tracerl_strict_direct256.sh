#!/usr/bin/env bash
set -Eeuo pipefail

# D13 is an iterative training-only correction from the D12b direct rollouts.
# It keeps every rollout and changes only the reward target: all-metal and
# charge-invalid plans are negative, charge-neutral Pauling-plausible plans are
# positive. Sampling remains direct generation with no candidate pool, rejection,
# retry, verifier selection, or distribution prior.

SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260530_2301-r5d12b-plantracerl-direct256}"
PLAN_NAME="${PLAN_NAME:-r5d13_countfields_tracerl_strict}"
REWARD_MODE="${REWARD_MODE:-strict_v2}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-runs/${SOURCE_RUN_ID}/outputs/r5d12b_countfields_tracerl_trace_rl/final}"
ROLLOUT_RAW_JSONL="${ROLLOUT_RAW_JSONL:-runs/${SOURCE_RUN_ID}/outputs/r5d12b_countfields_tracerl_plan_sample256_direct/raw_generations.jsonl}"
TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-240}"
TRACE_LR="${TRACE_LR:-6e-7}"
TRACE_BETA="${TRACE_BETA:-0.015}"

export SOURCE_RUN_ID PLAN_NAME REWARD_MODE BEST_CHECKPOINT ROLLOUT_RAW_JSONL
export TRACE_MAX_STEPS TRACE_LR TRACE_BETA

exec bash scripts/a800/run_r5d12_plan_tracerl_direct256.sh

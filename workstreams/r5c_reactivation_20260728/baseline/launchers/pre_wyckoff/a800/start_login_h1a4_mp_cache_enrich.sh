#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
PARENT_RUN_ID="${PARENT_RUN_ID:-20260604_h1a4_joint_basin_planner_clean}"
LOG_NAME="${LOG_NAME:-h1a4_login_cache_enrich_20260605.log}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${PARENT_RUN_ID}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/notes"

log_path="${RUN_DIR}/logs/${LOG_NAME}"
pid_path="${RUN_DIR}/notes/h1a4_login_cache_enrich_pid.txt"

nohup bash scripts/a800/login_h1a4_mp_cache_enrich.sh >"${log_path}" 2>&1 &
pid="$!"
echo "${pid}" >"${pid_path}"

echo "LOGIN_ENRICH_PID=${pid}"
echo "LOGIN_ENRICH_LOG=${log_path}"

#!/bin/bash
#SBATCH --job-name=v3-dlm-2g-assemble
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

: "${SMOKE_ARRAY_JOB_ID:?SMOKE_ARRAY_JOB_ID is required}"

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dlm_b1_b2_2xa800_smoke32_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SCHEDULER_RECORD="${RUN_ROOT}/scheduler_${SMOKE_ARRAY_JOB_ID}.txt"

test -d "${SOURCE_ROOT}"
test ! -e "${RUN_ROOT}/terminal_report.json"

sacct -n -X -j "${SMOKE_ARRAY_JOB_ID}" \
  --format=JobIDRaw,State,ExitCode,Elapsed -P > "${SCHEDULER_RECORD}"

"${PYTHON}" "${EXECUTION_DIR}/assemble_terminal.py" \
  --run-root "${RUN_ROOT}" \
  --array-job-id "${SMOKE_ARRAY_JOB_ID}" \
  --scheduler-record "${SCHEDULER_RECORD}" \
  --output "${RUN_ROOT}/terminal_report.json"

echo "h1a2_dlm_2xa800_smoke_terminal_complete=true"
echo "h1a2_dlm_2xa800_smoke_automatic_downstream=false"

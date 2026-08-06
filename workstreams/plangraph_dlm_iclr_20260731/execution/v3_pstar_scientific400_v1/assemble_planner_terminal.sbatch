#!/bin/bash
#SBATCH --job-name=v3-pstar-select
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

: "${PLANNER_ARRAY_JOB_ID:?PLANNER_ARRAY_JOB_ID is required}"

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_pstar_scientific400_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -d "${SOURCE_ROOT}"
test ! -e "${RUN_ROOT}/terminal_report.json"
sacct -n -X -j "${PLANNER_ARRAY_JOB_ID}" \
  --format=JobID,State,ExitCode,Elapsed -P > "${RUN_ROOT}/scheduler_${PLANNER_ARRAY_JOB_ID}.txt"

"${PYTHON}" "${EXECUTION_DIR}/assemble_planner_terminal.py" \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/terminal_report.json"

echo "planner_scientific_terminal_complete=true"
echo "planner_scientific_automatic_downstream=false"

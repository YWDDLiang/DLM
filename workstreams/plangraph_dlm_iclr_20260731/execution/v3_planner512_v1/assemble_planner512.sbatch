#!/bin/bash
#SBATCH --job-name=v3-p512-assemble
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_planner512_v1/logs/%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_planner512_v1/logs/%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_v3_planner512_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_planner512_v1"
TRAINING_TERMINAL="${PROJECT_ROOT}/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1/terminal_report.json"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_MANIFEST_SHA256:?submission must export EXPECTED_MANIFEST_SHA256}"
: "${PLANNER_ARRAY_JOB_ID:?submission must export PLANNER_ARRAY_JOB_ID}"

mkdir -p "${RUN_ROOT}/status"
sacct -j "${PLANNER_ARRAY_JOB_ID}" \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList \
  --parsable2 --noheader > "${RUN_ROOT}/status/planner_array_sacct.txt" || true

test "$(sha256sum "${EXECUTION_DIR}/EXECUTION_MANIFEST.json" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c "${EXECUTION_DIR}/SOURCE_SHA256.txt"

"${PYTHON}" "${EXECUTION_DIR}/assemble_planner512.py" \
  --run-root "${RUN_ROOT}" \
  --planner-training-terminal "${TRAINING_TERMINAL}" \
  --output "${RUN_ROOT}/terminal_report.json"

echo "PLANNER512_ASSEMBLY_COMPLETE"

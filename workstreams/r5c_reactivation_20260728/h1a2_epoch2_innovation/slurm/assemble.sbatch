#!/usr/bin/env bash
#SBATCH --job-name=h1a2c-jc-select-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/20260729_h1a2c_jointchem_v1/logs/%x-%j.out
#SBATCH --error=runs/20260729_h1a2c_jointchem_v1/logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
WORK_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation"
RUN_ROOT="${PROJECT_ROOT}/runs/20260729_h1a2c_jointchem_v1"
PYTHON="${PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
PATCH_SHA256="${H1A2C_JOINTCHEM_PATCH_SHA256:?H1A2C_JOINTCHEM_PATCH_SHA256 is required}"
EPOCH2_ADAPTER_SHA256="65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/status"
trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/assembly_exit_status.txt"; exit "${status}"' EXIT

test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test -s "${RUN_ROOT}/arms/P0/plan_report.json"
(cd "${WORK_ROOT}" && test "$(sha256sum SOURCE_SHA256.txt | awk '{print $1}')" = "${PATCH_SHA256}")
(cd "${WORK_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"

"${PYTHON}" "${WORK_ROOT}/code/assemble_jointchem.py" \
  --run-root "${RUN_ROOT}" \
  --execution-manifest-sha256 "${PATCH_SHA256}" \
  --initial-adapter-sha256 "${EPOCH2_ADAPTER_SHA256}" \
  --output "${RUN_ROOT}/terminal_report.json"

test -s "${RUN_ROOT}/terminal_report.json"
echo "SELECTION_REPORT_READY=${RUN_ROOT}/terminal_report.json"

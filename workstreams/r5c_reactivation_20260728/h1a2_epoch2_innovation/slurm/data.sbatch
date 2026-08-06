#!/usr/bin/env bash
#SBATCH --job-name=h1a2c-jc-data-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/20260729_h1a2c_jointchem_v1/logs/%x-%j.out
#SBATCH --error=runs/20260729_h1a2c_jointchem_v1/logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
WORK_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation"
RESTORED_BASELINE_ROOT="${PROJECT_ROOT}/workstreams/r5c_reactivation_20260728/baseline"
if [ -d "${RESTORED_BASELINE_ROOT}/crystal_dlm" ]; then
  RUNTIME_ROOT="${RESTORED_BASELINE_ROOT}"
else
  RUNTIME_ROOT="${PROJECT_ROOT}"
fi
RUN_ROOT="${PROJECT_ROOT}/runs/20260729_h1a2c_jointchem_v1"
PYTHON="${PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
SOURCE_DIR="${PROJECT_ROOT}/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base"
OUT_DIR="${RUN_ROOT}/data"
PATCH_SHA256="${H1A2C_JOINTCHEM_PATCH_SHA256:?H1A2C_JOINTCHEM_PATCH_SHA256 is required}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/data_exit_status.txt"; exit "${status}"' EXIT

test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test -x "${PYTHON}"
test -d "${SOURCE_DIR}"
test ! -e "${OUT_DIR}"
(cd "${WORK_ROOT}" && test "$(sha256sum SOURCE_SHA256.txt | awk '{print $1}')" = "${PATCH_SHA256}")
(cd "${WORK_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
(cd "${RUNTIME_ROOT}" && sha256sum -c "${WORK_ROOT}/RUNTIME_REQUIRED_SHA256.txt")

export PYTHONPATH="${WORK_ROOT}/code:${RUNTIME_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" -m py_compile "${WORK_ROOT}"/code/*.py "${WORK_ROOT}"/tests/*.py
"${PYTHON}" -m unittest discover -s "${WORK_ROOT}/tests" -p 'test_*.py' -v
"${PYTHON}" "${WORK_ROOT}/code/jointchem_data.py" \
  --input-dir "${SOURCE_DIR}" \
  --output-dir "${OUT_DIR}" \
  --seed 17 \
  --train-stream-rows 3200 \
  --positive-fraction 0.8 \
  --validation-rows 1024 \
  --max-negative-candidates 256 \
  --execution-manifest-sha256 "${PATCH_SHA256}"

test -s "${OUT_DIR}/_SUCCESS"
test -s "${OUT_DIR}/manifest.json"
echo "DATA_ACCEPTANCE=PASS"

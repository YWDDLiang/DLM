#!/bin/bash
#SBATCH --job-name=pg-dlm-eng32-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_engineering32_v1/slurm/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_engineering32_v1/slurm/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260731_plangraph_dlm_engineering32_v1"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/preflight_source_v2"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
PILOT_ROOT="${EXECUTION_ROOT}/engineering_pilot_32_v1"
MANIFEST="${EXECUTION_ROOT}/ENGINEERING_PILOT_32_MANIFEST_V1.json"
AUTHORIZATION="${PILOT_ROOT}/ENGINEERING_PILOT_32_AUTHORIZATION_V1.json"
RUNNER="${PILOT_ROOT}/run_engineering_pilot.py"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

MANIFEST_SHA256=097943f6947ca166a07cb8dcbce0b68b938e52ba1e2d850ec6cca45e0aa69777
AUTHORIZATION_SHA256=f5d368e4604fafae3a9022af4d8d83d5925ee09fb3a1f75f166011e203775e2e
RUNNER_SHA256=175f33c3c3a9b4b1f3ec1d36ecddf127846bef3d1a74449ffe2494a22835057a

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_MIN:-0}" -ne 0 ] \
  || [ "${SLURM_ARRAY_TASK_MAX:-3}" -ne 3 ] \
  || [ "${SLURM_ARRAY_TASK_STEP:-1}" -ne 1 ] \
  || [ "${SLURM_ARRAY_TASK_COUNT:-4}" -ne 4 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 3 ]; then
  echo "plangraph engineering32 Slurm resource identity changed" >&2
  exit 2
fi

cd "${SOURCE_ROOT}"
test -d "${RUN_ROOT}"
test -d "${RUN_ROOT}/slurm"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${AUTHORIZATION_SHA256}"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = "${RUNNER_SHA256}"

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN

GPU_CSV="${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 2 > "${GPU_CSV}" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

"${PYTHON}" "${RUNNER}" \
  --manifest "${MANIFEST}" \
  --manifest-sha256 "${MANIFEST_SHA256}" \
  --authorization "${AUTHORIZATION}" \
  --authorization-sha256 "${AUTHORIZATION_SHA256}" \
  --runner-sha256 "${RUNNER_SHA256}" \
  --arm-index "${SLURM_ARRAY_TASK_ID}"

echo "plangraph_engineering32_array_task_complete=${SLURM_ARRAY_TASK_ID}"
echo "plangraph_engineering32_automatic_downstream=false"

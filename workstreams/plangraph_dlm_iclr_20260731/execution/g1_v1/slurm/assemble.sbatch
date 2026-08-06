#!/usr/bin/env bash
#SBATCH --job-name=pg-g1-assemble-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_g1_v1/slurm/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_plangraph_dlm_g1_v1/slurm/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260731_plangraph_dlm_g1_v1"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/g1_source_v1"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
MANIFEST="${EXECUTION_ROOT}/G1_EXECUTION_MANIFEST_V1.json"
AUTHORIZATION="${EXECUTION_ROOT}/G1_AUTHORIZATION_V1.json"
INITIAL_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"
trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/assembly_exit_status.txt"; exit "${status}"' EXIT
test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/assembly"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" "${SOURCE_ROOT}/assemble_g1.py" \
  --run-root "${RUN_ROOT}" \
  --execution-manifest-sha256 "${G1_EXECUTION_MANIFEST_SHA256}" \
  --initial-adapter-sha256 "${INITIAL_ADAPTER_SHA256}" \
  --max-jsd-bits 0.15 \
  --max-shortcut-rate-delta 0.05

test -s "${RUN_ROOT}/terminal_report.json"
test -s "${RUN_ROOT}/decision.json"
echo "G1_ASSEMBLY_ACCEPTANCE=PASS"


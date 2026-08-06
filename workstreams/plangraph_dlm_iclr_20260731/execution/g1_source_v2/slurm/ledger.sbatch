#!/usr/bin/env bash
#SBATCH --job-name=pg-g1-ledger-v1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
DATASET_ROOT="${PROJECT_ROOT}/data/dlm_sft/mp_20_plangraph_v1_20260731"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"

trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/ledger_exit_status.txt"; exit "${status}"' EXIT
test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
test -d "${DATASET_ROOT}/planner"
test ! -e "${RUN_ROOT}/data"

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/ledger"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

"${PYTHON}" "${SOURCE_ROOT}/build_g1_ledger.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${RUN_ROOT}/data" \
  --train-sha256 1fc12bdaf90d81851e70a3b484e08214368084601eed852685771661c14f594c \
  --val-sha256 9a12174df84fd1a2bdf6b7ff4c0f637567415efc243fb04c321eae9cf421b085 \
  --train-rows 27136 \
  --val-rows 9047 \
  --attempts 512 \
  --seed-base 20260731

test -s "${RUN_ROOT}/data/ledger_report.json"
test -s "${RUN_ROOT}/data/ledger/seed_ledger.jsonl"
echo "G1_LEDGER_ACCEPTANCE=PASS"


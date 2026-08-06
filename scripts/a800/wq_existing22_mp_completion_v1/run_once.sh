#!/usr/bin/env bash
set -Eeuo pipefail
set -o noclobber

ROOT="/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"
PYTHON="/public/home/jiaosz/ywliang/ai4s/.venvs/mp_api_0_45_13_emmet0_85_1_py310_v4_system/bin/python"
CONTRACT="${ROOT}/configs/experiments/wyckoff_codiffusion/wq_existing22_mp_completion_v1.json"
CLAIM="${ROOT}/runs/20260720_0401-crysllmgen-wq-final-v3/notes/wq_existing22_mp_completion_v1_claim.json"
OUTPUT="${ROOT}/runs/20260720_0401-crysllmgen-wq-final-v3/outputs/wq_existing22_mp_completion_v1"
STDOUT="${ROOT}/runs/20260720_0401-crysllmgen-wq-final-v3/logs/wq-existing22-mp-completion-v1.out"
STDERR="${ROOT}/runs/20260720_0401-crysllmgen-wq-final-v3/logs/wq-existing22-mp-completion-v1.err"

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_NAME:-}" ]]; then
  echo "ERROR: login-node-only MP completion cannot run inside Slurm" >&2
  exit 2
fi
if [[ -z "${EXECUTION_PATCH_SHA256:-}" ]]; then
  echo "ERROR: EXECUTION_PATCH_SHA256 is required" >&2
  exit 3
fi
if [[ ! "${EXECUTION_PATCH_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: EXECUTION_PATCH_SHA256 is not a lowercase SHA256" >&2
  exit 4
fi
if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "ERROR: MP_API_KEY must be present in the process environment" >&2
  exit 5
fi
if [[ ! -x "${PYTHON}" || ! -f "${CONTRACT}" ]]; then
  echo "ERROR: frozen sidecar Python or contract is missing" >&2
  exit 6
fi
if [[ -e "${CLAIM}" || -e "${OUTPUT}" || -e "${STDOUT}" || -e "${STDERR}" ]]; then
  echo "ERROR: fixed claim/output/log identity already exists" >&2
  exit 7
fi

mkdir -p "$(dirname "${CLAIM}")" "$(dirname "${OUTPUT}")" "$(dirname "${STDOUT}")"
umask 077
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"

exec "${PYTHON}" diagnostics/complete_wq_existing22_mp_hull.py \
  --project-root "${ROOT}" \
  --contract "${CONTRACT}" \
  --execution-patch-sha256 "${EXECUTION_PATCH_SHA256}" \
  >"${STDOUT}" 2>"${STDERR}"

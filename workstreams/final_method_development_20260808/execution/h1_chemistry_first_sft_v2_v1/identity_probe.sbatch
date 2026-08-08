#!/bin/bash
#SBATCH --job-name=h1-p0-id-probe-v7
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7/logs/%j_identity_probe.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7/logs/%j_identity_probe.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7"
SOURCE_ROOT="${RUN_ROOT}/source"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${LEGACY_PYTHON:?missing runtime}"
test "${SLURM_JOB_PARTITION:-}" = gpu
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test ! -e "${RUN_ROOT}/probe/real_p0_identity_report.json"
mkdir -p "${RUN_ROOT}/probe" "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/identity_probe_exit_code.txt"
  touch "${RUN_ROOT}/status/identity_probe_FAILED"
  exit "${rc}"
}
trap on_error ERR

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/identity_probe_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${LEGACY_PYTHON}" scripts/a800/probe_h1_peft_protected_identity_v1.py \
  --model-path "${MODEL_PATH}" \
  --p0-adapter-path "${P0_ADAPTER}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/probe/real_p0_identity_report.json"
"${LEGACY_PYTHON}" scripts/a800/validate_h1_peft_identity_gate_v1.py probe \
  --report "${RUN_ROOT}/probe/real_p0_identity_report.json" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/probe/real_p0_identity_gate.json"
chmod 400 \
  "${RUN_ROOT}/probe/real_p0_identity_report.json" \
  "${RUN_ROOT}/probe/real_p0_identity_gate.json"
sha256sum "${RUN_ROOT}/probe/real_p0_identity_report.json" \
  > "${RUN_ROOT}/probe/real_p0_identity_report.sha256"
sha256sum "${RUN_ROOT}/probe/real_p0_identity_gate.json" \
  > "${RUN_ROOT}/probe/real_p0_identity_gate.sha256"
printf '0\n' > "${RUN_ROOT}/status/identity_probe_exit_code.txt"
touch "${RUN_ROOT}/status/identity_probe_SUCCESS"
echo "H1_PROTECTED_P0_IDENTITY_PROBE_PASS"

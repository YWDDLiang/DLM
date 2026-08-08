#!/bin/bash
#SBATCH --job-name=h1-cf-data-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_identity_repair_v4/logs/%j_data.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_identity_repair_v4/logs/%j_data.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_identity_repair_v4"
SOURCE_ROOT="${RUN_ROOT}/source"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
FROZEN_WITNESS_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact4_witness_input_v1"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${LEGACY_PYTHON:?missing legacy runtime}"
: "${LOCAL_SMACT4_WITNESS_ROOT:?missing local SMACT4 witness root}"
: "${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256:?missing local witness manifest identity}"
test "${SLURM_JOB_PARTITION:-}" = normal
test -x "${LEGACY_PYTHON}"
test "${LOCAL_SMACT4_WITNESS_ROOT}" = "${FROZEN_WITNESS_ROOT}"
test -f "${LOCAL_SMACT4_WITNESS_ROOT}/MANIFEST.json"
test -f "${LOCAL_SMACT4_WITNESS_ROOT}/_SUCCESS"
test -f "${RUN_ROOT}/status/snapshot_SUCCESS"
test -f "${RUN_ROOT}/legacy_snapshot/_SUCCESS"
test ! -e "${RUN_ROOT}/data"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir "${RUN_ROOT}/data"
mkdir -p "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/data_exit_code.txt"
  touch "${RUN_ROOT}/status/data_FAILED"
  exit "${rc}"
}
trap on_error ERR

"${LEGACY_PYTHON}" scripts/build_h1_chemistry_first_sft_data.py \
  --legacy-snapshot-dir "${RUN_ROOT}/legacy_snapshot" \
  --smact4-witness-bundle-dir "${LOCAL_SMACT4_WITNESS_ROOT}" \
  --expected-smact4-witness-manifest-sha256 "${EXPECTED_LOCAL_SMACT4_WITNESS_MANIFEST_SHA256}" \
  --expected-source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  --output-dir "${RUN_ROOT}/data/sft" \
  --tokenizer-path "${MODEL_PATH}" \
  --max-length 512 --seed 26080817

test -f "${RUN_ROOT}/data/sft/_SUCCESS"
test -f "${RUN_ROOT}/data/sft/audit_report.json"
test -f "${RUN_ROOT}/data/sft/ORDER_LEDGER.json"
sha256sum \
  "${RUN_ROOT}/legacy_snapshot/report.json" \
  "${LOCAL_SMACT4_WITNESS_ROOT}/MANIFEST.json" \
  "${RUN_ROOT}/data/sft/audit_report.json" \
  "${RUN_ROOT}/data/sft/ORDER_LEDGER.json" \
  "${RUN_ROOT}/data/sft/sft_v2/train.jsonl" \
  "${RUN_ROOT}/data/sft/sft_v2_c/train.jsonl" \
  > "${RUN_ROOT}/data/DATA_SHA256.txt"
printf '0\n' > "${RUN_ROOT}/status/data_exit_code.txt"
touch "${RUN_ROOT}/status/data_SUCCESS"
echo H1_CHEMISTRY_FIRST_DATA_PASS

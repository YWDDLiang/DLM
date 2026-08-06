#!/bin/bash
#SBATCH --job-name=h1-nc-data-v1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%j_data.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%j_data.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260806_h1_nocharge_ion_aux_sft_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_nocharge_ion_aux_sft_v1"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source inventory identity}"
: "${LEGACY_PYTHON:?missing legacy evaluator runtime}"
: "${SMACT4_PYTHON:?missing exact SMACT4 runtime}"
test "${SLURM_JOB_PARTITION:-}" = normal
test -x "${LEGACY_PYTHON}"
test -x "${SMACT4_PYTHON}"
test ! -e "${RUN_ROOT}/data"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

mkdir "${RUN_ROOT}/data"
mkdir -p "${RUN_ROOT}/status"
trap 'rc=$?; printf "%s\n" "$rc" > "${RUN_ROOT}/status/data_exit_code.txt"; touch "${RUN_ROOT}/status/data_FAILED"; exit "$rc"' ERR

"${LEGACY_PYTHON}" scripts/export_h1_nocharge_mp20_legacy_snapshot.py \
  --input-dir "${MP20_DIR}" \
  --output-dir "${RUN_ROOT}/data/legacy_snapshot"

"${SMACT4_PYTHON}" scripts/build_h1_nocharge_ion_aux_sft_data.py \
  --input-dir "${MP20_DIR}" \
  --legacy-snapshot-dir "${RUN_ROOT}/data/legacy_snapshot" \
  --output-dir "${RUN_ROOT}/data/sft" \
  --max-length 512 \
  --seed 26080617

"${LEGACY_PYTHON}" scripts/audit_h1_nocharge_sft_tokenizer.py \
  --data-root "${RUN_ROOT}/data/sft" \
  --tokenizer-path "${MODEL_PATH}" \
  --max-length 512 \
  --output-json "${RUN_ROOT}/data/tokenizer_audit.json"

sha256sum \
  "${RUN_ROOT}/data/legacy_snapshot/report.json" \
  "${RUN_ROOT}/data/sft/audit_report.json" \
  "${RUN_ROOT}/data/tokenizer_audit.json" \
  > "${RUN_ROOT}/data/DATA_SHA256.txt"
printf '0\n' > "${RUN_ROOT}/status/data_exit_code.txt"
touch "${RUN_ROOT}/status/data_SUCCESS"
echo H1_NOCHARGE_DATA_PASS

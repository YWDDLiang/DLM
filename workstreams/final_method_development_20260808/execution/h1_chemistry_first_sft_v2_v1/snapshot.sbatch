#!/bin/bash
#SBATCH --job-name=h1-cf-snapshot-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7/logs/%j_snapshot.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7/logs/%j_snapshot.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7"
SOURCE_ROOT="${RUN_ROOT}/source"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${LEGACY_PYTHON:?missing legacy runtime}"
test "${SLURM_JOB_PARTITION:-}" = normal
test -x "${LEGACY_PYTHON}"
test ! -e "${RUN_ROOT}/legacy_snapshot"
test ! -e "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export USE_TORCH=0 USE_TF=0 USE_FLAX=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/snapshot_exit_code.txt"
  touch "${RUN_ROOT}/status/snapshot_FAILED"
  exit "${rc}"
}
trap on_error ERR

"${LEGACY_PYTHON}" scripts/export_h1_nocharge_mp20_legacy_snapshot.py \
  --input-dir "${MP20_DIR}" \
  --output-dir "${RUN_ROOT}/legacy_snapshot"

test -f "${RUN_ROOT}/legacy_snapshot/_SUCCESS"
test -f "${RUN_ROOT}/legacy_snapshot/report.json"
for split in train val test; do
  test -f "${RUN_ROOT}/legacy_snapshot/${split}.jsonl"
done
tar -C "${RUN_ROOT}" -czf "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz.tmp" legacy_snapshot
mv "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz.tmp" "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz"
sha256sum \
  "${RUN_ROOT}/legacy_snapshot/report.json" \
  "${RUN_ROOT}/legacy_snapshot/train.jsonl" \
  "${RUN_ROOT}/legacy_snapshot/val.jsonl" \
  "${RUN_ROOT}/legacy_snapshot/test.jsonl" \
  "${RUN_ROOT}/legacy_snapshot_bundle.tar.gz" \
  > "${RUN_ROOT}/legacy_snapshot/SNAPSHOT_SHA256.txt"
printf '0\n' > "${RUN_ROOT}/status/snapshot_exit_code.txt"
touch "${RUN_ROOT}/status/snapshot_SUCCESS"
echo H1_CHEMISTRY_FIRST_LEGACY_SNAPSHOT_PASS

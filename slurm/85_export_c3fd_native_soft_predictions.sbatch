#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-soft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=c3fd-native-soft-%j.out
#SBATCH --error=c3fd-native-soft-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD17="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly C3FD18="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed18/checkpoint.pt"
readonly OUTPUT="${ROOT}/data/c3fd_native_soft_predictions_v1_20260830"
readonly RUN="${ROOT}/runs/export_c3fd_native_soft_predictions_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly C3FD17_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly C3FD18_SHA=d2e31996d034ba0f452ba72906539d6067bf3d7bf8969940c0f7961011b02b07
readonly TRAIN_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly VAL_SHA=67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly MANIFEST_SHA=9a494be1a70086af7a8c78eef40fafef8f34eb34d852e51311bf19f44d6586fd
readonly EXPORTER_SHA=03ba95cc664ce9d016bcc0e7bde8eed75d0a6ffe775f3c272a764e00de68841c

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${OUTPUT}.preparing" ]]; then
    mv "${OUTPUT}.preparing" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ "$(sha256sum "${C3FD17}" | awk '{print $1}')" == "${C3FD17_SHA}" ]] || exit 2
[[ "$(sha256sum "${C3FD18}" | awk '{print $1}')" == "${C3FD18_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/train.jsonl" | awk '{print $1}')" == "${TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/val.jsonl" | awk '{print $1}')" == "${VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')" == "${VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/manifest.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/export_c3fd_native_soft_predictions.py" | awk '{print $1}')" == "${EXPORTER_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_soft_predictions_export_v1\n'
  printf 'checkpoint_seed17=%s\n' "${C3FD17_SHA}"
  printf 'checkpoint_seed18=%s\n' "${C3FD18_SHA}"
  printf 'train=%s\n' "${TRAIN_SHA}"
  printf 'val=%s\n' "${VAL_SHA}"
  printf 'vocabulary=%s\n' "${VOCAB_SHA}"
  printf 'exporter=%s\n' "${EXPORTER_SHA}"
  printf 'device=cpu\n'
  printf 'checkpoint_selection=none\n'
  printf 'fields=lattice_system,spacegroup_bucket,volume_per_atom_bin\n'
  printf 'outcomes_read=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir -p "${ROOT}/runs/contracts"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_soft_predictions_export_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'checkpoints\tseed17,seed18\n'
  printf 'checkpoint_selection\tnone\n'
  printf 'device\tcpu\n'
  printf 'cpus\t16\n'
  printf 'outcomes_read\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${C3FD17}" "${C3FD18}" "${DATA}/train.jsonl" \
  "${DATA}/val.jsonl" "${DATA}/vocabulary.json" "${DATA}/manifest.json" \
  "${ROOT}/scripts/export_c3fd_native_soft_predictions.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${ROOT}/scripts/export_c3fd_native_soft_predictions.py" \
  --checkpoint "seed17=${C3FD17}" \
  --checkpoint "seed18=${C3FD18}" \
  --semantic-data-dir "${DATA}" \
  --batch-size 256 \
  --output-dir "${OUTPUT}" \
  >"${RUN}/export.out" 2>"${RUN}/export.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
[[ -f "${OUTPUT}/manifest.json" ]] || exit 3
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["selection"] == "none"; assert x["all_frozen_checkpoints_preserved"] is True; assert sorted(x["expected_checkpoint_seeds"]) == [17,18]; assert x["outcomes_read"] is False; assert x["splits"]["train"]["eligible_prediction_coverage"] == 1.0; assert x["splits"]["val"]["eligible_prediction_coverage"] == 1.0' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

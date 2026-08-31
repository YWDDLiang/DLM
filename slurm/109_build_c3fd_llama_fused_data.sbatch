#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-fused-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=c3fd-llama-fused-data-%j.out
#SBATCH --error=c3fd-llama-fused-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SEMANTIC="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly CTV="${ROOT}/data/ctv_minimal_spec_v4_20260828"
readonly OUTPUT="${ROOT}/data/c3fd_llama_fused_typed_v2_20260831"
readonly RUN="${ROOT}/runs/build_c3fd_llama_fused_data_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly BUILDER="${ROOT}/scripts/build_c3fd_llama_fused_data.py"
readonly CORE="${ROOT}/src/crystal_dlm/c3fd_llama_fused_plan.py"
readonly SEMANTIC_TRAIN_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly SEMANTIC_VAL_SHA=67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3
readonly SEMANTIC_VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly CTV_TRAIN_SHA="${C3FD_LLAMA_CTV_TRAIN_SHA256:?pin CTV train SHA256}"
readonly CTV_VAL_SHA="${C3FD_LLAMA_CTV_VAL_SHA256:?pin CTV val SHA256}"
readonly BUILDER_SHA="${C3FD_LLAMA_FUSED_BUILDER_SHA256:?pin builder SHA256}"
readonly CORE_SHA="${C3FD_LLAMA_FUSED_PLAN_SHA256:?pin fused-plan SHA256}"

mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ ! -e "${OUTPUT}" ]]
for spec in \
  "${SEMANTIC_TRAIN_SHA}:${SEMANTIC}/train.jsonl" \
  "${SEMANTIC_VAL_SHA}:${SEMANTIC}/val.jsonl" \
  "${SEMANTIC_VOCAB_SHA}:${SEMANTIC}/vocabulary.json" \
  "${CTV_TRAIN_SHA}:${CTV}/train.jsonl" \
  "${CTV_VAL_SHA}:${CTV}/val.jsonl" \
  "${BUILDER_SHA}:${BUILDER}" "${CORE_SHA}:${CORE}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
sha256sum "${SEMANTIC}/train.jsonl" "${SEMANTIC}/val.jsonl" \
  "${SEMANTIC}/vocabulary.json" "${CTV}/train.jsonl" "${CTV}/val.jsonl" \
  "${BUILDER}" "${CORE}" >"${RUN}/INPUTS.sha256"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "${PYTHON}" "${BUILDER}" --semantic-dir "${SEMANTIC}" \
  --ctv-minimal-dir "${CTV}" --output-dir "${OUTPUT}" \
  >"${RUN}/builder.out" 2>"${RUN}/builder.err"
[[ -f "${OUTPUT}/_SUCCESS" ]]
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="complete"; assert x["included_splits"]==["train","val"]; assert x["dev_test_prospective_rows"]==0; assert x["ordinal_zip_used"] is False; assert x["splits"]["train"]["kept_rows"]>0 and x["splits"]["val"]["kept_rows"]>0' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"

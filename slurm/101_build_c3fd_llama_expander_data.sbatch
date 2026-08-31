#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=c3fd-llama-data-%j.out
#SBATCH --error=c3fd-llama-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SEMANTIC="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly PREDICTED="${ROOT}/data/c3fd_native_soft_predictions_v1_20260830"
readonly OUTPUT="${ROOT}/data/c3fd_llama_rich_expander_v1_20260831"
readonly RUN="${ROOT}/runs/build_c3fd_llama_expander_data_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SEMANTIC_TRAIN_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly SEMANTIC_VAL_SHA=67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3
readonly SEMANTIC_VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly PREDICTED_TRAIN_SHA=967b4f7f1953e551a81d20197f22c5cb857ea31e14d7ec02ea4d240046d6c476
readonly PREDICTED_VAL_SHA=cf07f652bdc834f052572b7ef5103d8e3ac96a51a61b2008858ce9f06c07415b
readonly PREDICTED_MANIFEST_SHA=55a7141e8e864390a5af3702f1fc3f68d746e02838c733ff5a842cdcb9d0b84a
readonly BUILDER_SHA=f1cfd851e4fb4b9278964e5304b255638b988e8cdea5ff84cb3d31af2be9c6e3
readonly EXPANDER_SHA=a6a824118728c07ea6179613a0d295181ebb783471f80b0eb603f9a77da83007

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]] || exit 2
for spec in \
  "${SEMANTIC_TRAIN_SHA}:${SEMANTIC}/train.jsonl" \
  "${SEMANTIC_VAL_SHA}:${SEMANTIC}/val.jsonl" \
  "${SEMANTIC_VOCAB_SHA}:${SEMANTIC}/vocabulary.json" \
  "${PREDICTED_TRAIN_SHA}:${PREDICTED}/train.jsonl" \
  "${PREDICTED_VAL_SHA}:${PREDICTED}/val.jsonl" \
  "${PREDICTED_MANIFEST_SHA}:${PREDICTED}/manifest.json" \
  "${BUILDER_SHA}:${ROOT}/scripts/build_c3fd_llama_expander_data.py" \
  "${EXPANDER_SHA}:${ROOT}/src/crystal_dlm/c3fd_rich_expander.py"; do
  expected="${spec%%:*}"
  path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_rich_expander_data_v1
routes	F_formula_only,M_learned_soft_prefix
train_rows_expected	27136
validation_rows_expected	9047
feature_dim	54
predicted_checkpoints	seed17,seed18_no_selection
teacher_rich_fields	target_only
outcomes_read	false
EOF
sha256sum \
  "${SEMANTIC}/train.jsonl" "${SEMANTIC}/val.jsonl" "${SEMANTIC}/vocabulary.json" \
  "${PREDICTED}/train.jsonl" "${PREDICTED}/val.jsonl" "${PREDICTED}/manifest.json" \
  "${ROOT}/scripts/build_c3fd_llama_expander_data.py" \
  "${ROOT}/src/crystal_dlm/c3fd_rich_expander.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${PYTHON}" "${ROOT}/scripts/build_c3fd_llama_expander_data.py" \
  --semantic-dir "${SEMANTIC}" --predicted-dir "${PREDICTED}" \
  --output-dir "${OUTPUT}" >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" - "${OUTPUT}/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
assert x["routes"] == ["F", "M"] and x["feature_dim"] == 54
assert x["outcomes_read"] is False and x["predicted_checkpoint_selection"].startswith("none")
assert x["splits"]["train"]["source_rows"] == 27136
assert x["splits"]["val"]["source_rows"] == 9047
assert all(x["splits"][s]["visible_answers_identical"] for s in ("train","val"))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

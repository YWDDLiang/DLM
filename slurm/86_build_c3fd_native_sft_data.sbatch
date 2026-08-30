#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=c3fd-native-data-%j.out
#SBATCH --error=c3fd-native-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly INPUT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length
readonly SEMANTIC="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly PREDICTED="${ROOT}/data/c3fd_native_soft_predictions_v1_20260830"
readonly OUTPUT="${ROOT}/data/c3fd_native_sft_v1_20260830"
readonly RUN="${ROOT}/runs/build_c3fd_native_sft_data_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly INPUT_TRAIN_SHA=7e96c28b6bbfde8658c5c09f060b389292df322497712f78cfab6689dcfbd590
readonly INPUT_VAL_SHA=81c8d21a4942410835a31f00fa81b550c400584a11ae654a67207015d1f845a2
readonly INPUT_VOCAB_SHA=71d3145913eb3496c82f28c654989e37b0031105f52291483788aca3a8bee56b
readonly SEMANTIC_TRAIN_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly SEMANTIC_VAL_SHA=67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3
readonly SEMANTIC_VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly PREDICTED_TRAIN_SHA=967b4f7f1953e551a81d20197f22c5cb857ea31e14d7ec02ea4d240046d6c476
readonly PREDICTED_VAL_SHA=cf07f652bdc834f052572b7ef5103d8e3ac96a51a61b2008858ce9f06c07415b
readonly PREDICTED_MANIFEST_SHA=55a7141e8e864390a5af3702f1fc3f68d746e02838c733ff5a842cdcb9d0b84a
readonly BUILDER_SHA=1014f026ac7789156c74f80329c4f96ae87b22e1adce9a1347ad1606bdcede14
readonly NATIVE_PLAN_SHA=deb73a68a1757e35c14ed520365c33df6245dead0d8b30742d550ea5918105ad

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
[[ "$(sha256sum "${INPUT}/train.jsonl" | awk '{print $1}')" == "${INPUT_TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${INPUT}/val.jsonl" | awk '{print $1}')" == "${INPUT_VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${INPUT}/vocab_tokens.txt" | awk '{print $1}')" == "${INPUT_VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${SEMANTIC}/train.jsonl" | awk '{print $1}')" == "${SEMANTIC_TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${SEMANTIC}/val.jsonl" | awk '{print $1}')" == "${SEMANTIC_VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${SEMANTIC}/vocabulary.json" | awk '{print $1}')" == "${SEMANTIC_VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${PREDICTED}/train.jsonl" | awk '{print $1}')" == "${PREDICTED_TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${PREDICTED}/val.jsonl" | awk '{print $1}')" == "${PREDICTED_VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${PREDICTED}/manifest.json" | awk '{print $1}')" == "${PREDICTED_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/build_c3fd_native_sft_data.py" | awk '{print $1}')" == "${BUILDER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/crystal_dlm/c3fd_native_plan.py" | awk '{print $1}')" == "${NATIVE_PLAN_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_sft_data_build_v1\n'
  printf 'input_train=%s\n' "${INPUT_TRAIN_SHA}"
  printf 'input_val=%s\n' "${INPUT_VAL_SHA}"
  printf 'semantic_train=%s\n' "${SEMANTIC_TRAIN_SHA}"
  printf 'semantic_val=%s\n' "${SEMANTIC_VAL_SHA}"
  printf 'predicted_train=%s\n' "${PREDICTED_TRAIN_SHA}"
  printf 'predicted_val=%s\n' "${PREDICTED_VAL_SHA}"
  printf 'builder=%s\n' "${BUILDER_SHA}"
  printf 'native_plan=%s\n' "${NATIVE_PLAN_SHA}"
  printf 'views=teacher-native,predicted-native-seed17,predicted-native-seed18,soft-masked,minimal-reference\n'
  printf 'planner_checkpoint_selection=none\n'
  printf 'outcomes_read=false\n'
  printf 'split_policy=mp20-standard\n'
  printf 'expected_train_val_chemsys_overlap=3469\n'
  printf 'source_index_policy=file_ordinal_if_absent\n'
  printf 'dlm_interface=N_elements_counts_family_LS_SG_VP\n'
  printf 'planner_certificates_exposed=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir -p "${ROOT}/runs/contracts"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_sft_data_build_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'views\t5\n'
  printf 'planner_checkpoint_selection\tnone\n'
  printf 'device\tcpu\n'
  printf 'outcomes_read\tfalse\n'
  printf 'legacy_single_prediction\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${INPUT}/train.jsonl" "${INPUT}/val.jsonl" "${INPUT}/vocab_tokens.txt" \
  "${SEMANTIC}/train.jsonl" "${SEMANTIC}/val.jsonl" \
  "${SEMANTIC}/vocabulary.json" "${PREDICTED}/train.jsonl" \
  "${PREDICTED}/val.jsonl" "${PREDICTED}/manifest.json" \
  "${ROOT}/scripts/build_c3fd_native_sft_data.py" \
  "${ROOT}/src/crystal_dlm/c3fd_native_plan.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${PYTHON}" "${ROOT}/scripts/build_c3fd_native_sft_data.py" \
  --input-dir "${INPUT}" \
  --semantic-dir "${SEMANTIC}" \
  --predicted-soft-dir "${PREDICTED}" \
  --split-policy mp20-standard \
  --allow-source-ordinal-index \
  --output-dir "${OUTPUT}" \
  >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); expected=["teacher-native","predicted-native-seed17","predicted-native-seed18","soft-masked","minimal-reference"]; assert x["prediction_mode"] == "formal-multi-checkpoint"; assert x["prediction_checkpoint_order"] == ["seed17","seed18"]; assert x["views"] == expected; assert x["legacy_single_prediction_development"] is False; assert x["split_policy"] == "mp20-standard"; assert x["chemsys_held_out"] is False; assert x["chemsys_overlap"]["train__val"] == 3469; assert all(x["gate"].values()); assert x["splits"]["train"]["source_rows"] == 27136; assert x["splits"]["val"]["source_rows"] == 9047; assert x["splits"]["train"]["source_rows_assigned_ordinal_index"] == 27136; assert x["splits"]["val"]["source_rows_assigned_ordinal_index"] == 9047; assert x["splits"]["train"]["output_rows"] == 5*x["splits"]["train"]["source_rows"]; assert x["splits"]["val"]["output_rows"] == 5*x["splits"]["val"]["source_rows"]; assert x["gate"]["planner_certificates_not_exposed_to_dlm"] is True' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

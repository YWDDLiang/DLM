#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-align-groups
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=c3fd-native-align-groups-%j.out
#SBATCH --error=c3fd-native-align-groups-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly COHORT="${ROOT}/cohorts/c3fd_native_alignment_pool_train32x2xK4_v1_20260831"
readonly GENERATION="${ROOT}/runs/c3fd_native_alignment_pool_generation_38857"
readonly EVAL="${ROOT}/runs/c3fd_native_alignment_pool_eval_38881"
readonly OUTPUT="${ROOT}/data/c3fd_native_alignment_groups_v1_20260831"
readonly RUN="${ROOT}/runs/build_c3fd_native_alignment_groups_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly COHORT_MANIFEST_SHA=6c603ec1e86dbf904760460aac7e1b44581d518f7bbb2a712e9cafd597fe50d9
readonly GROUPS_SHA=72f2a08a704b211144f7874b9e1f74454082e2548c68e6055a5fdc720a4f792c
readonly GENERATION_FINAL_SHA=ff5bb27e2dbdf2c80c1af5ac950c1a8323167ba3bb884729a97218229c1d6e91
readonly BUILDER_SHA=97d5b2342a39065d8913be6aa60f7c3f9d17b503779d52e1744babd938926a3e

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${OUTPUT}" && ! -f "${OUTPUT}/_SUCCESS" ]]; then
    mv "${OUTPUT}" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ -f "${COHORT}/_SUCCESS" && -f "${GENERATION}/_SUCCESS" && -f "${EVAL}/_OFFLINE_SUCCESS" ]] || exit 2
[[ "$(sha256sum "${COHORT}/manifest.json" | awk '{print $1}')" == "${COHORT_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/groups.jsonl" | awk '{print $1}')" == "${GROUPS_SHA}" ]] || exit 2
[[ "$(sha256sum "${GENERATION}/C3FD_NATIVE_ALIGNMENT_POOL_GENERATION_FINAL.json" | awk '{print $1}')" == "${GENERATION_FINAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/build_c3fd_native_alignment_groups.py" | awk '{print $1}')" == "${BUILDER_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_alignment_group_build_v1\n'
  printf 'cohort=%s\n' "${COHORT_MANIFEST_SHA}"
  printf 'groups=%s\n' "${GROUPS_SHA}"
  printf 'generation=%s\n' "${GENERATION_FINAL_SHA}"
  printf 'eval_outputs=%s\n' "$(sha256sum "${EVAL}/OUTPUTS.sha256" | awk '{print $1}')"
  printf 'builder=%s\n' "${BUILDER_SHA}"
  printf 'policies=82017,82018\n'
  printf 'groups_per_policy=64\n'
  printf 'K=4\n'
  printf 'old_3614=false\n'
  printf 'prospective_outcomes=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir -p "${ROOT}/runs/contracts"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_alignment_group_build_v1\n'
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'policies\t82017,82018\n'
  printf 'groups_per_policy\t64\n'
  printf 'K\t4\n'
  printf 'raw_invalid_policy\tlexicographic_worst\n'
  printf 'valid_order\tsame_group_refined_CHGNet\n'
  printf 'best_anchor\tbest_raw_valid_refined_energy\n'
  printf 'group_weight\t1\n'
  printf 'cross_composition_energy\tfalse\n'
  printf 'old_3614_used\tfalse\n'
  printf 'prospective_outcomes_read\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${COHORT}/manifest.json" "${COHORT}/pool_plans.jsonl" \
  "${COHORT}/groups.jsonl" \
  "${GENERATION}/C3FD_NATIVE_ALIGNMENT_POOL_GENERATION_FINAL.json" \
  "${EVAL}/OUTPUTS.sha256" "${ROOT}/scripts/build_c3fd_native_alignment_groups.py" \
  >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${ROOT}/scripts/build_c3fd_native_alignment_groups.py" \
  --cohort "${COHORT}" --generation-run "${GENERATION}" --eval-run "${EVAL}" \
  --output-dir "${OUTPUT}" >"${RUN}/build.out" 2>"${RUN}/build.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["source"] == "fresh_MP20_train_on_policy_pool_only"; assert x["policies"] == [82017,82018]; assert x["groups_per_policy"] == 64; assert x["candidates_per_group"] == 4; assert x["group_weight"] == 1.0; assert x["historical_3614_used"] is False; assert x["prospective_outcomes_read"] is False; assert all(x["gates"].values()); assert all(x["audit"][str(p)]["trainable_groups"] > 0 for p in (82017,82018))' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

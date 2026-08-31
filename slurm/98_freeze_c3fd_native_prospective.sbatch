#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-prosp-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=c3fd-native-prosp-freeze-%j.out
#SBATCH --error=c3fd-native-prosp-freeze-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE_RUN="${ROOT}/runs/c3fd_native_prospective_seed20_38903"
readonly SOURCE="${SOURCE_RUN}/plans/plans_for_dlm.jsonl"
readonly MP20="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/train.jsonl"
readonly OUTPUT="${ROOT}/cohorts/c3fd_native_prospective_seed20_256_v1_20260831"
readonly RUN="${ROOT}/runs/freeze_c3fd_native_prospective_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SOURCE_SHA=1cbdb46128ea2ee62d924f4cec0c94fcd450e82b309535c317056ecbd63371df
readonly MP20_SHA=4cc292cb551cda3356a403debf4a81b4dbb82eb7a7e0db259c2d537b78efba51
readonly BUILDER_SHA=b8e1afe56e626dd2fd15b391cfdeebee950cca42ec0781cba5b703bac827bb55

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
[[ -f "${SOURCE_RUN}/_SUCCESS" ]] || exit 2
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] || exit 2
[[ "$(sha256sum "${MP20}" | awk '{print $1}')" == "${MP20_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" | awk '{print $1}')" == "${BUILDER_SHA}" ]] || exit 2
find "${ROOT}/cohorts" -type f -name '*.jsonl' -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/COHORT_EXCLUSIONS_BEFORE.sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_prospective_freeze_v1\n'
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'planner_sampling_seed\t20\n'
  printf 'source_sha256\t%s\n' "${SOURCE_SHA}"
  printf 'selection\tfirst_eligible_source_ordinal\n'
  printf 'selected\t256\n'
  printf 'blocked\tMP20_train_and_all_existing_cohorts_exact_identity\n'
  printf 'prompt_schema\tC3FD_NATIVE_PLAN_V2\n'
  printf 'outcomes_read\tfalse\n'
  printf 'policy_outcomes_exist_at_freeze\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${SOURCE}" "${MP20}" \
  "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" >"${RUN}/INPUTS.sha256"
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  --source-plans "${SOURCE}" --mp20-train "${MP20}" \
  --exclude-cohort-root "${ROOT}/cohorts" --output-dir "${OUTPUT}" --count 256 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"
[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
(cd "${OUTPUT}" && sha256sum -c SHA256SUMS)
"${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["planner_sampling_seed"] == 20; assert x["selected"] == 256; assert x["outcomes_read"] is False; assert x["policy_outcomes_exist_at_freeze"] is False; assert all(x["gates"].values())' "${OUTPUT}/manifest.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

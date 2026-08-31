#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=c3fd-llama-freeze-%j.out
#SBATCH --error=c3fd-llama-freeze-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CANARY_RUN="${C3FD_LLAMA_CANARY_RUN:?export completed canary run}"
readonly SOURCE_RUN="${ROOT}/runs/c3fd_prospective_seed21_38936"
readonly SOURCE_PLANS="${SOURCE_RUN}/plans/plans_for_dlm.jsonl"
readonly SOURCE_RAW="${SOURCE_RUN}/plans/raw_generations.jsonl"
readonly MP20_TRAIN=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length/train.jsonl
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD17="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly C3FD18="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed18/checkpoint.pt"
readonly COHORT="${ROOT}/cohorts/c3fd_llama_prospective_seed21_256_v1_20260831"
readonly CONDITIONS="${ROOT}/data/c3fd_llama_prospective_conditions_seed21_256_v1_20260831"
readonly RUN="${ROOT}/runs/freeze_c3fd_llama_prospective_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly C3FD17_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly C3FD18_SHA=d2e31996d034ba0f452ba72906539d6067bf3d7bf8969940c0f7961011b02b07
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly FREEZER_SHA=e3477a4292cafa6633c264a0962406143afbc4cd8172074330ed186ec14bd484
readonly MATERIALIZER_SHA=a71061a7b6ca6ae288bd18b8ea4525a7e535f20f464ba591a21e66694a5af354
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

[[ -f "${CANARY_RUN}/_SUCCESS" ]]
[[ -f "${SOURCE_RUN}/_SUCCESS" ]]
[[ ! -e "${COHORT}" && ! -e "${CONDITIONS}" ]]
for spec in \
  "${C3FD17_SHA}:${C3FD17}" "${C3FD18_SHA}:${C3FD18}" \
  "${VOCAB_SHA}:${DATA}/vocabulary.json" \
  "${FREEZER_SHA}:${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  "${MATERIALIZER_SHA}:${ROOT}/scripts/materialize_c3fd_llama_prospective_conditions.py" \
  "${EXPANDER_SHA}:${ROOT}/src/crystal_dlm/c3fd_rich_expander.py"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_prospective_freeze_v1
source_sampling_seed	21
selection	first256_eligible_source_ordinals
blocked	MP20_train_and_all_existing_cohorts
conditions	F_formula_only,M_seed17_seed18_soft_prefix
canary_dependency	${CANARY_RUN}
policy_or_DLM_outcomes_read	false
EOF

"${PYTHON}" "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  --source-plans "${SOURCE_PLANS}" --mp20-train "${MP20_TRAIN}" \
  --exclude-cohort-root "${ROOT}/cohorts" --output-dir "${COHORT}" \
  --count 256 --planner-sampling-seed 21 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

"${PYTHON}" "${ROOT}/scripts/materialize_c3fd_llama_prospective_conditions.py" \
  --source-raw "${SOURCE_RAW}" --ledger "${COHORT}/ledger.jsonl" \
  --data-dir "${DATA}" --checkpoint "seed17=${C3FD17}" \
  --checkpoint "seed18=${C3FD18}" --output-dir "${CONDITIONS}" \
  >"${RUN}/materialize.out" 2>"${RUN}/materialize.err"

[[ -f "${COHORT}/_SUCCESS" && -f "${CONDITIONS}/_SUCCESS" ]]
[[ "$(wc -l <"${CONDITIONS}/F.jsonl")" -eq 256 ]]
[[ "$(wc -l <"${CONDITIONS}/M.jsonl")" -eq 256 ]]
"${PYTHON}" - "${COHORT}/manifest.json" "${CONDITIONS}/manifest.json" <<'PY'
import json,sys
cohort=json.load(open(sys.argv[1],encoding="utf-8")); conditions=json.load(open(sys.argv[2],encoding="utf-8"))
assert cohort['planner_sampling_seed']==21 and cohort['selected']==256
assert all(cohort['gates'].values()) and cohort['outcomes_read'] is False
assert conditions['selected']==256 and conditions['routes']==['F','M']
assert conditions['checkpoint_order']==['seed17','seed18'] and conditions['outcomes_read'] is False
PY

sha256sum "${CANARY_RUN}/_SUCCESS" "${SOURCE_PLANS}" "${SOURCE_RAW}" \
  "${MP20_TRAIN}" "${COHORT}/manifest.json" "${COHORT}/ledger.jsonl" \
  "${CONDITIONS}/manifest.json" "${CONDITIONS}/F.jsonl" "${CONDITIONS}/M.jsonl" \
  "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  "${ROOT}/scripts/materialize_c3fd_llama_prospective_conditions.py" \
  >"${RUN}/INPUTS_OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n%s\n' "${COHORT}" "${CONDITIONS}"

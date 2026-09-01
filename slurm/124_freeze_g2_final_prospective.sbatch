#!/usr/bin/env bash
#SBATCH --job-name=freeze-g2-final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=freeze-g2-final-%j.out
#SBATCH --error=freeze-g2-final-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE_RUN="${G2_FINAL_SOURCE_RUN:?export completed seed22 source run}"
readonly SOURCE_PLANS="${SOURCE_RUN}/plans/plans_for_dlm.jsonl"
readonly MP20_TRAIN=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length/train.jsonl
readonly COHORT="${ROOT}/cohorts/c3fd_llama_g2_final_seed22_256_v1_20260901"
readonly RUN="${ROOT}/runs/freeze_c3fd_llama_g2_final_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly FREEZER_SHA=e3477a4292cafa6633c264a0962406143afbc4cd8172074330ed186ec14bd484

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ -f "${SOURCE_RUN}/_SUCCESS" && ! -e "${COHORT}" ]]
[[ "$(sha256sum "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py"|awk '{print $1}')" == "${FREEZER_SHA}" ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_g2_final_prospective_freeze_v1
source_sampling_seed	22
selection	first256_eligible_source_ordinals
blocked	MP20_train_and_all_existing_cohorts
policy_or_DLM_outcomes_read	false
method_frozen_before_cohort	G2_periodic_relation
EOF
"${PYTHON}" "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  --source-plans "${SOURCE_PLANS}" --mp20-train "${MP20_TRAIN}" \
  --exclude-cohort-root "${ROOT}/cohorts" --output-dir "${COHORT}" \
  --count 256 --planner-sampling-seed 22 >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"
[[ -f "${COHORT}/_SUCCESS" ]]
"${PYTHON}" - "${COHORT}/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['planner_sampling_seed']==22 and x['selected']==256
assert all(x['gates'].values()) and x['outcomes_read'] is False
PY
sha256sum "${SOURCE_PLANS}" "${MP20_TRAIN}" "${COHORT}/manifest.json" "${COHORT}/ledger.jsonl" \
  "${ROOT}/scripts/freeze_c3fd_native_prospective_cohort.py" >"${RUN}/INPUTS_OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${COHORT}"

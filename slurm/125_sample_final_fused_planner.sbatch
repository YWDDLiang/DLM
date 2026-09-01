#!/usr/bin/env bash
#SBATCH --job-name=final-fused-plan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=final-fused-plan-%j.out
#SBATCH --error=final-fused-plan-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly LEDGER="${ROOT}/cohorts/c3fd_llama_g2_final_seed22_256_v1_20260901/ledger.jsonl"
readonly FUSED="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly LLAMA=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly SAMPLER="${CODE_ROOT}/src/scripts/sample_c3fd_llama_typed_planner.py"
readonly RUN="${ROOT}/runs/c3fd_llama_g2_final_typed_plans_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly LEDGER_SHA=90109e1b5398bba679ad11b92bfadc85f050af44f758f6cbc024bfe88234e842
readonly C3FD_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly TYPED_CONFIG_SHA=ca786e3d78f737e840ae29aed9e7a61f316ada193fb4a6c385e2941cefbff89a
readonly TYPED_STATE_SHA=9912e2594834df9412ab7d7cf6e3d5fc88258c5b0c707b5677503668958871bc
readonly ADAPTER_CONFIG_SHA=e9e6da2ce613c1d78e504eadec701d2a93e671a32e8801fa0a0a78c79dcaf8d5
readonly ADAPTER_MODEL_SHA=7638b05d56e7ed19c20229849aa26c75d55cf618264ab8fe2e205c2195884c75
readonly SAMPLER_SHA=0c7fcfd041343d6cb0d39564bb4c1dcda989b70f4e74b0c2270cffc2fede01c5

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
actual_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${CODE_COMMIT}" ]]
for spec in "${LEDGER_SHA}:${LEDGER}" "${C3FD_SHA}:${C3FD}" "${VOCAB_SHA}:${DATA}/vocabulary.json" \
 "${TYPED_CONFIG_SHA}:${FUSED}/typed_residual_config.json" "${TYPED_STATE_SHA}:${FUSED}/typed_residual_state.pt" \
 "${ADAPTER_CONFIG_SHA}:${FUSED}/llama_adapter/adapter_config.json" "${ADAPTER_MODEL_SHA}:${FUSED}/llama_adapter/adapter_model.safetensors" \
 "${SAMPLER_SHA}:${SAMPLER}"; do expected="${spec%%:*}"; path="${spec#*:}"; [[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]]; done
[[ -f "$(dirname "${LEDGER}")/_SUCCESS" && -f "${FUSED}/_SUCCESS" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"; [[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_g2_final_typed_plans_v1
source_seed	22
source_ledger_sha256	${LEDGER_SHA}
planner	frozen_job39051
sampling_seed	22
requested	256
temperature	0.9
top_p	0.95
trajectory_per_ordinal	1
minimum_comp_valid	0.95
outcomes_read	false
retry_filter_replacement_rerank_best_of_n	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${LEDGER}" "${C3FD}" "${DATA}/vocabulary.json" \
 "${FUSED}/typed_residual_config.json" "${FUSED}/typed_residual_state.pt" \
 "${FUSED}/llama_adapter/adapter_config.json" "${FUSED}/llama_adapter/adapter_model.safetensors" "${SAMPLER}" >"${RUN}/INPUTS.sha256"
env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
 "${PYTHON}" "${SAMPLER}" --c3fd-checkpoint "${C3FD}" --data-dir "${DATA}" \
 --llama-model "${LLAMA}" --fused-planner-final "${FUSED}" --source-ledger "${LEDGER}" \
 --output-dir "${OUTPUT}" --expected-c3fd-sha256 "${C3FD_SHA}" --expected-vocabulary-sha256 "${VOCAB_SHA}" \
 --expected-source-ledger-sha256 "${LEDGER_SHA}" --expected-typed-config-sha256 "${TYPED_CONFIG_SHA}" \
 --expected-typed-state-sha256 "${TYPED_STATE_SHA}" --expected-adapter-config-sha256 "${ADAPTER_CONFIG_SHA}" \
 --expected-adapter-model-sha256 "${ADAPTER_MODEL_SHA}" --requested 256 --seed 22 --expected-seed 22 --temperature 0.9 \
 --top-p 0.95 --top-k 0 --max-species 7 --minimum-comp-valid 0.95 >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"
[[ -f "${OUTPUT}/_SUCCESS" && "$(wc -l <"${OUTPUT}/raw_generations.jsonl")" -eq 256 ]]
"${PYTHON}" - "${OUTPUT}/sample_metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['requested_samples']==256 and x['one_trajectory_per_ordinal'] is True
assert x['comp_valid_rate_requested_denominator']>=0.95
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

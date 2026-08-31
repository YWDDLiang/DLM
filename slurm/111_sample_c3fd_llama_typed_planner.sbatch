#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-typed-sample
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=c3fd-llama-typed-sample-%j.out
#SBATCH --error=c3fd-llama-typed-sample-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE_LEDGER="${C3FD_LLAMA_TYPED_SOURCE_LEDGER:?export frozen source ledger}"
readonly SOURCE_LEDGER_SHA="${C3FD_LLAMA_TYPED_SOURCE_LEDGER_SHA256:?export pinned ledger SHA256}"
readonly FUSED_FINAL="${C3FD_LLAMA_TYPED_FINAL:?export completed fused Planner final}"
readonly TYPED_CONFIG_SHA="${C3FD_LLAMA_TYPED_CONFIG_SHA256:?export pinned typed config SHA256}"
readonly TYPED_STATE_SHA="${C3FD_LLAMA_TYPED_STATE_SHA256:?export pinned typed state SHA256}"
readonly ADAPTER_CONFIG_SHA="${C3FD_LLAMA_TYPED_ADAPTER_CONFIG_SHA256:?export pinned adapter config SHA256}"
readonly ADAPTER_MODEL_SHA="${C3FD_LLAMA_TYPED_ADAPTER_MODEL_SHA256:?export pinned adapter model SHA256}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD_CHECKPOINT="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly LLAMA_MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly SAMPLER="${ROOT}/src/scripts/sample_c3fd_llama_typed_planner.py"
readonly RUN="${ROOT}/runs/c3fd_llama_typed_plans_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly C3FD_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly SAMPLER_SHA=35bd60c7de186cd4cd2fa5c86a9f9c5da32732bfe3216a435a186ee0c901b55f

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

[[ -f "$(dirname "${SOURCE_LEDGER}")/_SUCCESS" ]]
[[ -f "${FUSED_FINAL}/_SUCCESS" ]]
for spec in \
  "${C3FD_SHA}:${C3FD_CHECKPOINT}" \
  "${VOCAB_SHA}:${DATA}/vocabulary.json" \
  "${SOURCE_LEDGER_SHA}:${SOURCE_LEDGER}" \
  "${TYPED_CONFIG_SHA}:${FUSED_FINAL}/typed_residual_config.json" \
  "${TYPED_STATE_SHA}:${FUSED_FINAL}/typed_residual_state.pt" \
  "${ADAPTER_CONFIG_SHA}:${FUSED_FINAL}/llama_adapter/adapter_config.json" \
  "${ADAPTER_MODEL_SHA}:${FUSED_FINAL}/llama_adapter/adapter_model.safetensors" \
  "${SAMPLER_SHA}:${SAMPLER}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose its GPU}"
[[ "${#allocated_gpus[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_typed_single_trajectory_slurm111_v1
c3fd_checkpoint	v2.5_seed17_frozen
fused_planner	fresh_llama_lora_plus_typed_residual_final
source_ledger	${SOURCE_LEDGER}
source_ledger_sha256	${SOURCE_LEDGER_SHA}
stability_goal	meta_or_better
sampling_seed	21
requested	256
temperature	0.9
top_p	0.95
top_k	0
trajectory_per_ordinal	1
proposal_pre_mask	pauling_bitset_can_complete_family_N_arity
terminal_eos	exact_benchmark_compatible_only
serializer	C3FD_NATIVE_PLAN_V2
minimum_comp_valid	0.95
retry_filter_replacement_rerank_best_of_n	false
EOF
sha256sum \
  "${C3FD_CHECKPOINT}" "${DATA}/vocabulary.json" "${SOURCE_LEDGER}" \
  "${FUSED_FINAL}/typed_residual_config.json" \
  "${FUSED_FINAL}/typed_residual_state.pt" \
  "${FUSED_FINAL}/llama_adapter/adapter_config.json" \
  "${FUSED_FINAL}/llama_adapter/adapter_model.safetensors" "${SAMPLER}" \
  >"${RUN}/INPUTS.sha256"

env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SAMPLER}" \
  --c3fd-checkpoint "${C3FD_CHECKPOINT}" --data-dir "${DATA}" \
  --llama-model "${LLAMA_MODEL}" --fused-planner-final "${FUSED_FINAL}" \
  --source-ledger "${SOURCE_LEDGER}" --output-dir "${OUTPUT}" \
  --expected-c3fd-sha256 "${C3FD_SHA}" \
  --expected-vocabulary-sha256 "${VOCAB_SHA}" \
  --expected-source-ledger-sha256 "${SOURCE_LEDGER_SHA}" \
  --expected-typed-config-sha256 "${TYPED_CONFIG_SHA}" \
  --expected-typed-state-sha256 "${TYPED_STATE_SHA}" \
  --expected-adapter-config-sha256 "${ADAPTER_CONFIG_SHA}" \
  --expected-adapter-model-sha256 "${ADAPTER_MODEL_SHA}" \
  --requested 256 --seed 21 --temperature 0.9 --top-p 0.95 --top-k 0 \
  --max-species 7 --minimum-comp-valid 0.95 \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/raw_generations.jsonl")" -eq 256 ]]
"${PYTHON}" - "${OUTPUT}/sample_metrics.json" <<'PY'
import json, pathlib, sys
metrics = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert metrics["requested_samples"] == 256
assert metrics["one_trajectory_per_ordinal"] is True
assert metrics["comp_valid_rate_requested_denominator"] >= 0.95
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

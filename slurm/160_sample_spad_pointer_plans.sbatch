#!/usr/bin/env bash
#SBATCH --job-name=spad-plans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=spad-plans-%j.out
#SBATCH --error=spad-plans-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly PLANNER="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly POINTER="${ROOT}/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt"
readonly LEDGER="${ROOT}/data/c3fd_llama_prospective_conditions_seed21_256_v1_20260831/F.jsonl"
readonly CONTROL="${ROOT}/runs/c3fd_llama_typed_plans_39088/plans/plans_for_dlm.jsonl"
readonly RUN="${ROOT}/runs/spad_pointer_plans_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${POINTER}" "${LEDGER}" "${CONTROL}" \
  "${PLANNER}/typed_residual_config.json" "${PLANNER}/typed_residual_state.pt" \
  "${PLANNER}/llama_adapter/adapter_config.json" \
  "${PLANNER}/llama_adapter/adapter_model.safetensors" \
  "${SOURCE}/src/scripts/sample_c3fd_llama_typed_planner.py"; do
  [[ -f "${path}" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/sample_c3fd_llama_typed_planner.py" \
  --c3fd-checkpoint "${C3FD}" --data-dir "${DATA}" \
  --llama-model "${MODEL}" --fused-planner-final "${PLANNER}" \
  --species-pointer-state "${POINTER}" --source-ledger "${LEDGER}" \
  --output-dir "${OUTPUT}" \
  --expected-c3fd-sha256 87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53 \
  --expected-vocabulary-sha256 3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3 \
  --expected-source-ledger-sha256 0b1f2c13bd47af63c23db2c63f939c871e04b9e9761c6f2cc714c4a4abe32417 \
  --expected-typed-config-sha256 ca786e3d78f737e840ae29aed9e7a61f316ada193fb4a6c385e2941cefbff89a \
  --expected-typed-state-sha256 9912e2594834df9412ab7d7cf6e3d5fc88258c5b0c707b5677503668958871bc \
  --expected-adapter-config-sha256 e9e6da2ce613c1d78e504eadec701d2a93e671a32e8801fa0a0a78c79dcaf8d5 \
  --expected-adapter-model-sha256 7638b05d56e7ed19c20229849aa26c75d55cf618264ab8fe2e205c2195884c75 \
  --requested 256 --seed 21 --temperature 0.9 --top-p 0.95 --top-k 0 \
  --max-species 7 --minimum-comp-valid 0.95 \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
"${PYTHON}" - "${OUTPUT}/plans_for_dlm.jsonl" "${CONTROL}" "${RUN}" <<'PY'
import json,pathlib,sys
new=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
old=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
assert len(new)==len(old)==256
assert [x['sample_idx'] for x in new]==list(range(256))
assert [x['plan_text'] for x in new]==[x['plan_text'] for x in old]
assert all(x['species_program_source']=='planner_llama_pointer' for x in new)
assert all(sorted(x['species_program_indices'])==list(range(len(x['species_program_indices']))) for x in new)
changed=sum(x['species_program_indices']!=list(range(len(x['species_program_indices']))) for x in new)
report={'requested':256,'plan_identity_vs_control':256,'pointer_programs':256,
        'noncanonical_programs':changed,'composition_mutation':False}
pathlib.Path(sys.argv[3],'SPAD_PLAN_FINAL.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

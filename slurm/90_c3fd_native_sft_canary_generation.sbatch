#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-sft-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-native-sft-canary-%j.out
#SBATCH --error=c3fd-native-sft-canary-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAIN_RUN="${ROOT}/runs/c3fd_native_teacher_sft_38703"
readonly POLICY17="${TRAIN_RUN}/train_seed82017/checkpoints/step-3392"
readonly POLICY18="${TRAIN_RUN}/train_seed82018/checkpoints/step-3392"
readonly COHORT="${ROOT}/cohorts/c3fd_native_sft_canary_train128_val128_v1_20260831"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/c3fd_native_sft_canary_generation_${SLURM_JOB_ID}"
readonly MANIFEST_SHA=73177562b54e30276af833ebffcf1a07082bc9a610d1099f28693b64a7a1b116
readonly LEDGER_SHA=3a6cae692af6f524e810f8597df28e5024a0cea06be3ecc48c92fb0eb6fb9816
readonly PLAN17_SHA=a1b82120ea6697800c4fd4b0857d64ad6c393b101f1f278384f15310db5c37a5
readonly PLAN18_SHA=586ccdda4ac84089bf731105c6ab1173daf29eae554a490d1c70564ee89128c9
readonly POLICY17_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20
readonly POLICY18_SHA=7f22580a78b4823b44ee841a25cce9fd39541bb0a2192171b0adb4788725526c
readonly POLICY_CONFIG_SHA=61ff08995286cf7679fb2e494064892a778a925becd2d55300a70e354f8ff240
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA=51e5bc0303778df9382052b2cae65d9922c74efd4136482788bd2d2ea0c44ee8
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498

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

for path in "${MODEL}" "${TRAIN_RUN}/_SUCCESS" "${COHORT}/_SUCCESS" \
  "${POLICY17}/adapter_model.safetensors" "${POLICY18}/adapter_model.safetensors" \
  "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${COHORT}/manifest.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/ledger.jsonl" | awk '{print $1}')" == "${LEDGER_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/planner_seed17.jsonl" | awk '{print $1}')" == "${PLAN17_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/planner_seed18.jsonl" | awk '{print $1}')" == "${PLAN18_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY17}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY17_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY18}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY18_SHA}" ]] || exit 2
for config in "${POLICY17}/adapter_config.json" "${POLICY18}/adapter_config.json"; do
  [[ "$(sha256sum "${config}" | awk '{print $1}')" == "${POLICY_CONFIG_SHA}" ]] || exit 2
done
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/sample_sgtc_l6.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_sft_canary_generation_v1\n'
  printf 'cohort_manifest=%s\n' "${MANIFEST_SHA}"
  printf 'ledger=%s\n' "${LEDGER_SHA}"
  printf 'planner_seed17=%s\n' "${PLAN17_SHA}"
  printf 'planner_seed18=%s\n' "${PLAN18_SHA}"
  printf 'policy82017=%s\n' "${POLICY17_SHA}"
  printf 'policy82018=%s\n' "${POLICY18_SHA}"
  printf 'sampler=%s\n' "${SAMPLER_SHA}"
  printf 'refiner_script=%s\n' "${REFINER_SCRIPT_SHA}"
  printf 'model494=%s\n' "${REFINER_SHA}"
  printf 'requested=256\n'
  printf 'split=train128,val128\n'
  printf 'policies=82017,82018\n'
  printf 'streams=17,18\n'
  printf 'planner_by_stream=17:seed17,18:seed18\n'
  printf 'temperature=0.7\n'
  printf 'tau=800\n'
  printf 'one_plan_one_trajectory=true\n'
  printf 'rerank_retry_replacement=false\n'
  printf 'official_query=false\n'
} >"${RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir -p "${ROOT}/runs/contracts"
mkdir "${CONTRACT_DIR}"
cp "${RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_sft_canary_generation_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'requested\t256\n'
  printf 'source_split\ttrain128,val128\n'
  printf 'policies\t82017,82018\n'
  printf 'streams\t17,18\n'
  printf 'planner_checkpoint_by_stream\tseed17,seed18\n'
  printf 'dlm_seeds\t91117,92117\n'
  printf 'refiner_seeds\t101117,102117\n'
  printf 'temperature\t0.7\n'
  printf 'schedule\texact_axis\n'
  printf 'refiner_tau\t800\n'
  printf 'attempts_per_plan\t1\n'
  printf 'seed_or_checkpoint_selection\tfalse\n'
  printf 'retry_rerank_replacement\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${COHORT}/manifest.json" "${COHORT}/ledger.jsonl" \
  "${COHORT}/planner_seed17.jsonl" "${COHORT}/planner_seed18.jsonl" \
  "${POLICY17}/adapter_config.json" "${POLICY17}/adapter_model.safetensors" \
  "${POLICY18}/adapter_config.json" "${POLICY18}/adapter_model.safetensors" \
  "${REFINER}" "${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" >"${RUN}/INPUTS.sha256"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 4 ]] || exit 2

run_cell() {
  local slot="$1" stream="$2" policy="$3" checkpoint="$4" plan_file="$5"
  local device dlm_seed refiner_seed output metrics output_file
  device="${GPU_DEVICES[${slot}]}"
  if [[ "${stream}" == 17 ]]; then
    dlm_seed=91117
    refiner_seed=101117
  else
    dlm_seed=92117
    refiner_seed=102117
  fi
  output="${RUN}/stream${stream}/policy${policy}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" --checkpoint-path "${checkpoint}" \
    --prompt-jsonl "${plan_file}" --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" --num-samples 256 --temperature 0.7 \
    --seed "${dlm_seed}" >"${output}/body.out" 2>"${output}/body.err"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${refiner_seed}" \
    --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${output}/refine.out" 2>"${output}/refine.err"
  [[ -f "${output}/body/_SUCCESS" ]] || exit 3
  metrics="${output}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]] || exit 3
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]] || exit 3
  touch "${output}/refine/_SUCCESS" "${output}/_SUCCESS"
}

run_cell 0 17 82017 "${POLICY17}" "${COHORT}/planner_seed17.jsonl" & p0=$!
run_cell 1 17 82018 "${POLICY18}" "${COHORT}/planner_seed17.jsonl" & p1=$!
run_cell 2 18 82017 "${POLICY17}" "${COHORT}/planner_seed18.jsonl" & p2=$!
run_cell 3 18 82018 "${POLICY18}" "${COHORT}/planner_seed18.jsonl" & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for stream in 17 18; do
  for policy in 82017 82018; do
    [[ -f "${RUN}/stream${stream}/policy${policy}/_SUCCESS" ]] || exit 3
  done
done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

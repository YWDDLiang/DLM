#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-align-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-native-align-gen-%j.out
#SBATCH --error=c3fd-native-align-gen-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAIN_RUN="${ROOT}/runs/c3fd_native_teacher_sft_38703"
readonly POLICY17="${TRAIN_RUN}/train_seed82017/checkpoints/step-3392"
readonly POLICY18="${TRAIN_RUN}/train_seed82018/checkpoints/step-3392"
readonly COHORT="${ROOT}/cohorts/c3fd_native_alignment_pool_train32x2xK4_v1_20260831"
readonly PLANS="${COHORT}/pool_plans.jsonl"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/c3fd_native_alignment_pool_generation_${SLURM_JOB_ID}"
readonly MANIFEST_SHA=6c603ec1e86dbf904760460aac7e1b44581d518f7bbb2a712e9cafd597fe50d9
readonly PLANS_SHA=8e03e9defff800f3df494b776182b313e9a2ec0af6d5587765071a7f091ffc95
readonly GROUPS_SHA=72f2a08a704b211144f7874b9e1f74454082e2548c68e6055a5fdc720a4f792c
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
[[ "$(sha256sum "${PLANS}" | awk '{print $1}')" == "${PLANS_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/groups.jsonl" | awk '{print $1}')" == "${GROUPS_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY17}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY17_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY18}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY18_SHA}" ]] || exit 2
for config in "${POLICY17}/adapter_config.json" "${POLICY18}/adapter_config.json"; do
  [[ "$(sha256sum "${config}" | awk '{print $1}')" == "${POLICY_CONFIG_SHA}" ]] || exit 2
done
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/sample_sgtc_l6.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2

{
  printf 'schema=c3fd_native_alignment_pool_generation_v1\n'
  printf 'manifest=%s\n' "${MANIFEST_SHA}"
  printf 'plans=%s\n' "${PLANS_SHA}"
  printf 'groups=%s\n' "${GROUPS_SHA}"
  printf 'policies=82017,82018\n'
  printf 'groups_per_policy=64\n'
  printf 'K=4\n'
  printf 'rows_per_policy=256\n'
  printf 'dlm_seed=93117\n'
  printf 'refiner_seed=103117\n'
  printf 'temperature=0.7\n'
  printf 'tau=800\n'
  printf 'retry_rerank_replacement=false\n'
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
  printf 'schema\tc3fd_native_alignment_pool_generation_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'source\tMP20_train_only\n'
  printf 'policies\t82017,82018\n'
  printf 'groups_per_policy\t64\n'
  printf 'candidates_per_group\t4\n'
  printf 'attempts_per_policy\t256\n'
  printf 'shared_dlm_seed\t93117\n'
  printf 'shared_refiner_seed\t103117\n'
  printf 'temperature\t0.7\n'
  printf 'schedule\texact_axis\n'
  printf 'refiner_tau\t800\n'
  printf 'retry_rerank_replacement\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${COHORT}/manifest.json" "${PLANS}" "${COHORT}/groups.jsonl" \
  "${POLICY17}/adapter_config.json" "${POLICY17}/adapter_model.safetensors" \
  "${POLICY18}/adapter_config.json" "${POLICY18}/adapter_model.safetensors" \
  "${REFINER}" "${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" >"${RUN}/INPUTS.sha256"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 2 ]] || exit 2

run_cell() {
  local slot="$1" policy="$2" checkpoint="$3"
  local device output metrics output_file
  device="${GPU_DEVICES[${slot}]}"
  output="${RUN}/policy${policy}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" --checkpoint-path "${checkpoint}" \
    --prompt-jsonl "${PLANS}" --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" --num-samples 256 --temperature 0.7 \
    --seed 93117 >"${output}/body.out" 2>"${output}/body.err"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed 103117 \
    --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${output}/refine.out" 2>"${output}/refine.err"
  [[ -f "${output}/body/_SUCCESS" ]] || exit 3
  metrics="${output}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]] || exit 3
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]] || exit 3
  touch "${output}/refine/_SUCCESS" "${output}/_SUCCESS"
}

run_cell 0 82017 "${POLICY17}" & p0=$!
run_cell 1 82018 "${POLICY18}" & p1=$!
wait "${p0}"
wait "${p1}"
for policy in 82017 82018; do
  [[ -f "${RUN}/policy${policy}/_SUCCESS" ]] || exit 3
done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-rich-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-rich-canary-%j.out
#SBATCH --error=h1a2-rich-canary-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly RUN="${ROOT}/runs/rich_recovery_canary_generation_${SLURM_JOB_ID}"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" \
    >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly RICH_DLM="${REPO}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
readonly MINIMAL_DLM="${ROOT}/runs/ctv_minimal_base_36898/train/checkpoints/step-696"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly COHORT="${ROOT}/cohorts/rich_recovery_dev_seed19_256_v2_20260830"
readonly MP20="${REPO}/reference/crysllmgen/data/mp_20"
readonly C3FD_DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD17="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly C3FD18="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed18/checkpoint.pt"
readonly LIVE_ASSET_MANIFEST="${ROOT}/data/rich_dlm_live_asset_manifest_v1_20260830/RICH_DLM_LIVE_ASSET_MANIFEST.json"
readonly CONTRACT_SHA=f35d62150d81b8698e17096a0a7538b3a809d83e9317d981ef94deedee32825d
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly MANIFEST_SHA=ad27cde7ea32549cdf6254b28f97181efab75017d594a490c7d827a2754c7735
readonly M0_SHA=e875ac961337be33fb7cbcc82fad9f93de71de5ce7c9a4c12429ddea5da02ac6
readonly R0_SHA=e40a00fa407fff384fbb991f75aece67d49ebb6d41867090a05bc3ca8140355c
readonly RCF_SHA=a161f9e89dfc7fb6ddfdc1649ac0a5296889aea161adbf7c1be0da36cd9876e7
readonly LEDGER_SHA=61b83c87ae79a79432b09578e8f6936c47731bfa46911631d9dc105b96649669
readonly LIVE_ASSET_MANIFEST_SHA=b2b591fa793d060a615c28c41f94900fe1aafc11c5cd76819b3f85cc7f269942
readonly LIVE_AUDITOR_SHA=620bc34b4e61f7164c830d2303fe4ee4305eca55192979a93daef2b9d75462a6
readonly CRYS_CODE_SHA=643458ab620fcaef0e8b037dca1add228a9c8d3558d11138b1f41fc4d6851691
readonly RICH_ADAPTER_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
readonly MINIMAL_ADAPTER_SHA=6ea3c2a633706968e4b3e3cf77e98e46399c23e1568333522283472634553ecb
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA=51e5bc0303778df9382052b2cae65d9922c74efd4136482788bd2d2ea0c44ee8
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498

mkdir -p "${ROOT}/runs/contracts"
{
  printf 'schema=h1a2_rich_recovery_canary_generation_v2\n'
  printf 'cohort_manifest=%s\n' "${MANIFEST_SHA}"
  printf 'M0=%s\n' "${M0_SHA}"
  printf 'R0=%s\n' "${R0_SHA}"
  printf 'RCF=%s\n' "${RCF_SHA}"
  printf 'ledger=%s\n' "${LEDGER_SHA}"
  printf 'live_assets=%s\n' "${LIVE_ASSET_MANIFEST_SHA}"
  printf 'live_auditor=%s\n' "${LIVE_AUDITOR_SHA}"
  printf 'crys_code=%s\n' "${CRYS_CODE_SHA}"
  printf 'rich_adapter=%s\n' "${RICH_ADAPTER_SHA}"
  printf 'minimal_adapter=%s\n' "${MINIMAL_ADAPTER_SHA}"
  printf 'model494=%s\n' "${REFINER_SHA}"
  printf 'sampler=%s\n' "${SAMPLER_SHA}"
  printf 'refiner=%s\n' "${REFINER_SCRIPT_SHA}"
  printf 'arms=M0,RCF,R0\n'
  printf 'streams=17,18\n'
  printf 'dlm_seeds=91117,92117\n'
  printf 'refiner_seeds=101117,102117\n'
  printf 'temperature=0.7\n'
  printf 'tau=800\n'
  printf 'denominator=256\n'
  printf 'rcf_value_changed_min=250\n'
  printf 'refined_missing=preserve_by_sample_idx\n'
  printf 'rerank=false\n'
  printf 'replacement=false\n'
} >"${RUN}/contract_payload.txt"
[[ "$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')" == "${CONTRACT_SHA}" ]] || exit 2

mkdir "${CONTRACT_DIR}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\t%s\n' 'h1a2_rich_recovery_canary_generation_v2'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'cohort\t%s\n' "${COHORT}"
  printf 'cohort_manifest_sha256\t%s\n' "${MANIFEST_SHA}"
  printf 'arms\tM0,RCF,R0\n'
  printf 'streams\t17,18\n'
  printf 'dlm_seeds\t91117,92117\n'
  printf 'refiner_seeds\t101117,102117\n'
  printf 'denominator\t256\n'
  printf 'temperature\t0.7\n'
  printf 'schedule\texact_axis\n'
  printf 'refiner_tau\t800\n'
  printf 'attempts_per_plan\t1\n'
  printf 'rcf_tuple_value_changed_min\t250\n'
  printf 'refined_missing_policy\tpreserve_by_sample_idx_in_offline_assembly\n'
  printf 'rerank_or_replacement\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

for path in \
  "${MODEL}" "${RICH_DLM}" "${MINIMAL_DLM}" "${REFINER}" "${CRYS}" \
  "${COHORT}/_SUCCESS" \
  "${ROOT}/data/rich_dlm_live_asset_manifest_v1_20260830/_SUCCESS"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${COHORT}/RICH_RECOVERY_COHORT_MANIFEST.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${LIVE_ASSET_MANIFEST}" | awk '{print $1}')" == "${LIVE_ASSET_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/audit_live_assets.py" | awk '{print $1}')" == "${LIVE_AUDITOR_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/M0.jsonl" | awk '{print $1}')" == "${M0_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/R0.jsonl" | awk '{print $1}')" == "${R0_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/RCF.jsonl" | awk '{print $1}')" == "${RCF_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/ledger.jsonl" | awk '{print $1}')" == "${LEDGER_SHA}" ]] || exit 2
[[ "$(sha256sum "${RICH_DLM}/adapter_model.safetensors" | awk '{print $1}')" == "${RICH_ADAPTER_SHA}" ]] || exit 2
[[ "$(sha256sum "${MINIMAL_DLM}/adapter_model.safetensors" | awk '{print $1}')" == "${MINIMAL_ADAPTER_SHA}" ]] || exit 2
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/sample_sgtc_l6.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2
python -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["gates"]["pass"] is True; assert int(x["selection"]["rcf_tuple_value_changed"]) >= 250' "${COHORT}/RICH_RECOVERY_COHORT_MANIFEST.json"

find "${CRYS}" -path "${CRYS}/out" -prune -o -path "${CRYS}/data" -prune \
  -o -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/CRYS_CODE_FILES.sha256"
[[ "$(sha256sum "${RUN}/CRYS_CODE_FILES.sha256" | awk '{print $1}')" == "${CRYS_CODE_SHA}" ]] || exit 2

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python "${ROOT}/scripts/audit_live_assets.py" \
  --asset "rich_dlm=${RICH_DLM}" \
  --asset "minimal_dlm=${MINIMAL_DLM}" \
  --asset "dlm_base=${MODEL}" \
  --asset "model494=${REFINER}" \
  --asset "mp20=${MP20}" \
  --asset "c3fd_checkpoint17=${C3FD17}" \
  --asset "c3fd_checkpoint18=${C3FD18}" \
  --asset "c3fd_data=${C3FD_DATA}" \
  --asset "body_sampler=${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
  --asset "rich_parser=${ROOT}/src/crystal_dlm/r5_plan_body.py" \
  --asset "c3fd_sampler=${ROOT}/scripts/sample_c3fd_plans.py" \
  --asset "refiner=${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --asset "evaluator=${ROOT}/eval_runtime/run_full_reconstructed_eval.py" \
  --expected-file "rich_dlm:adapter_model.safetensors=${RICH_ADAPTER_SHA}" \
  --workers 4 \
  --output-dir "${RUN}/live_assets_recheck" \
  >"${RUN}/live_assets_recheck.out" 2>"${RUN}/live_assets_recheck.err"
[[ "$(sha256sum "${RUN}/live_assets_recheck/RICH_DLM_LIVE_ASSET_MANIFEST.json" | awk '{print $1}')" == "${LIVE_ASSET_MANIFEST_SHA}" ]] || exit 2

sha256sum \
  "${COHORT}/RICH_RECOVERY_COHORT_MANIFEST.json" \
  "${LIVE_ASSET_MANIFEST}" \
  "${COHORT}/M0.jsonl" "${COHORT}/R0.jsonl" "${COHORT}/RCF.jsonl" \
  "${COHORT}/ledger.jsonl" \
  "${RICH_DLM}/adapter_model.safetensors" \
  "${MINIMAL_DLM}/adapter_model.safetensors" \
  "${REFINER}" \
  "${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  "${ROOT}/scripts/audit_live_assets.py" \
  >"${RUN}/INPUTS.sha256"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 6 ]] || exit 2

run_cell() {
  local gpu="$1" stream="$2" arm="$3" checkpoint="$4" view="$5"
  local device dlm_seed refiner_seed output
  device="${GPU_DEVICES[${gpu}]}"
  if [[ "${stream}" == 17 ]]; then
    dlm_seed=91117
    refiner_seed=101117
  else
    dlm_seed=92117
    refiner_seed=102117
  fi
  output="${RUN}/stream${stream}/${arm}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --prompt-jsonl "${COHORT}/${view}.jsonl" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --num-samples 256 \
    --temperature 0.7 \
    --seed "${dlm_seed}" \
    >"${output}/body.out" 2>"${output}/body.err"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed "${refiner_seed}" \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 256 \
    >"${output}/refine.out" 2>"${output}/refine.err"
  [[ -f "${output}/body/_SUCCESS" ]] || exit 3
  [[ -f "${output}/refine/_SUCCESS" ]] || exit 3
  touch "${output}/_SUCCESS"
}

run_cell 0 17 M0 "${MINIMAL_DLM}" M0 & p0=$!
run_cell 1 17 RCF "${RICH_DLM}" RCF & p1=$!
run_cell 2 17 R0 "${RICH_DLM}" R0 & p2=$!
run_cell 3 18 M0 "${MINIMAL_DLM}" M0 & p3=$!
run_cell 4 18 RCF "${RICH_DLM}" RCF & p4=$!
run_cell 5 18 R0 "${RICH_DLM}" R0 & p5=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"
wait "${p4}"
wait "${p5}"

for stream in 17 18; do
  for arm in M0 RCF R0; do
    [[ -f "${RUN}/stream${stream}/${arm}/_SUCCESS" ]] || exit 3
  done
done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

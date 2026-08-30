#!/usr/bin/env bash
#SBATCH --job-name=h1a2-faithful-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=h1a2-faithful-gen-%j.out
#SBATCH --error=h1a2-faithful-gen-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly RUN="${ROOT}/runs/faithful_rich_diagnostic_generation_${SLURM_JOB_ID}"
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
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly COHORT="${ROOT}/cohorts/faithful_rich_diagnostic_h0_r0s_256_v2_20260830"
readonly LIVE_ASSET_MANIFEST="${ROOT}/data/rich_dlm_live_asset_manifest_v1_20260830/RICH_DLM_LIVE_ASSET_MANIFEST.json"
readonly H0_SHA=fc370f7a38b6348a1450d6e9cbf969740cadb9cb4c10c581686bc235c70e4634
readonly R0S_SHA=2846c67acea80dbc5d2e17db93f4cbf8168d3851c81f57f945d379db8c10ad45
readonly LEDGER_SHA=406dbf48721bb528a3c954e06e59c930799cc72ca9d59ddb11e2eb911a6b5206
readonly MANIFEST_SHA=f1d002568aff0382f2cdd08c99d0cfec5a36d794bc7c02a709aee0036315b259
readonly LIVE_ASSET_MANIFEST_SHA=b2b591fa793d060a615c28c41f94900fe1aafc11c5cd76819b3f85cc7f269942
readonly CRYS_CODE_SHA=643458ab620fcaef0e8b037dca1add228a9c8d3558d11138b1f41fc4d6851691
readonly RICH_ADAPTER_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA=51e5bc0303778df9382052b2cae65d9922c74efd4136482788bd2d2ea0c44ee8
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498

for path in \
  "${MODEL}" "${RICH_DLM}" "${REFINER}" "${CRYS}" \
  "${COHORT}/_SUCCESS" \
  "${ROOT}/data/rich_dlm_live_asset_manifest_v1_20260830/_SUCCESS"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${COHORT}/H0.jsonl" | awk '{print $1}')" == "${H0_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/R0S.jsonl" | awk '{print $1}')" == "${R0S_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/DIAGNOSTIC_LEDGER.jsonl" | awk '{print $1}')" == "${LEDGER_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/FAITHFUL_RICH_DIAGNOSTIC_MANIFEST.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${LIVE_ASSET_MANIFEST}" | awk '{print $1}')" == "${LIVE_ASSET_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${RICH_DLM}/adapter_model.safetensors" | awk '{print $1}')" == "${RICH_ADAPTER_SHA}" ]] || exit 2
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/sample_sgtc_l6.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2

python -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); assert x["status"] == "frozen_outcome_blind_development_diagnostic"; assert x["outcome_fields_read"] == []; assert x["official_query_permitted"] is False; assert x["views"]["H0"]["rows"] == 256; assert x["views"]["R0S"]["rows"] == 256; assert x["views"]["R0S"]["exact_identity_mismatches"] == 0; assert x["views"]["R0S"]["soft_tuple_mismatches"] == 0' "${COHORT}/FAITHFUL_RICH_DIAGNOSTIC_MANIFEST.json"

find "${CRYS}" -path "${CRYS}/out" -prune -o -path "${CRYS}/data" -prune \
  -o -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/CRYS_CODE_FILES.sha256"
[[ "$(sha256sum "${RUN}/CRYS_CODE_FILES.sha256" | awk '{print $1}')" == "${CRYS_CODE_SHA}" ]] || exit 2

{
  printf 'schema=h1a2_faithful_rich_diagnostic_generation_v1\n'
  printf 'H0=%s\n' "${H0_SHA}"
  printf 'R0S=%s\n' "${R0S_SHA}"
  printf 'ledger=%s\n' "${LEDGER_SHA}"
  printf 'manifest=%s\n' "${MANIFEST_SHA}"
  printf 'rich_adapter=%s\n' "${RICH_ADAPTER_SHA}"
  printf 'model494=%s\n' "${REFINER_SHA}"
  printf 'sampler=%s\n' "${SAMPLER_SHA}"
  printf 'refiner=%s\n' "${REFINER_SCRIPT_SHA}"
  printf 'crys_code=%s\n' "${CRYS_CODE_SHA}"
  printf 'views=H0,R0S\n'
  printf 'streams=17,18\n'
  printf 'dlm_seeds=91117,92117\n'
  printf 'refiner_seeds=101117,102117\n'
  printf 'temperature=0.7\n'
  printf 'schedule=exact_axis\n'
  printf 'tau=800\n'
  printf 'denominator=256\n'
  printf 'attempts_per_plan=1\n'
  printf 'rerank=false\n'
  printf 'replacement=false\n'
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
  printf 'schema\th1a2_faithful_rich_diagnostic_generation_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'cohort\t%s\n' "${COHORT}"
  printf 'views\tH0,R0S\n'
  printf 'streams\t17,18\n'
  printf 'denominator\t256\n'
  printf 'temperature\t0.7\n'
  printf 'schedule\texact_axis\n'
  printf 'refiner_tau\t800\n'
  printf 'attempts_per_plan\t1\n'
  printf 'missing_policy\tpreserve_by_sample_idx\n'
  printf 'rerank_or_replacement\tfalse\n'
  printf 'development_diagnostic\ttrue\n'
  printf 'official_query\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

sha256sum \
  "${COHORT}/H0.jsonl" "${COHORT}/R0S.jsonl" \
  "${COHORT}/DIAGNOSTIC_LEDGER.jsonl" \
  "${COHORT}/FAITHFUL_RICH_DIAGNOSTIC_MANIFEST.json" \
  "${LIVE_ASSET_MANIFEST}" \
  "${RICH_DLM}/adapter_model.safetensors" "${REFINER}" \
  "${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  >"${RUN}/INPUTS.sha256"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 4 ]] || exit 2

run_cell() {
  local slot="$1" stream="$2" view="$3"
  local device dlm_seed refiner_seed output metrics output_file
  device="${GPU_DEVICES[${slot}]}"
  if [[ "${stream}" == 17 ]]; then
    dlm_seed=91117
    refiner_seed=101117
  else
    dlm_seed=92117
    refiner_seed=102117
  fi
  output="${RUN}/stream${stream}/${view}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${RICH_DLM}" \
    --prompt-jsonl "${COHORT}/${view}.jsonl" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --num-samples 256 \
    --temperature 0.7 \
    --seed "${dlm_seed}" \
    >"${output}/body.out" 2>"${output}/body.err"
  [[ -f "${output}/body/_SUCCESS" ]] || exit 3
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
  metrics="${output}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]] || exit 3
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]] || exit 3
  touch "${output}/refine/_SUCCESS"
  touch "${output}/_SUCCESS"
}

run_cell 0 17 H0 & p0=$!
run_cell 1 17 R0S & p1=$!
run_cell 2 18 H0 & p2=$!
run_cell 3 18 R0S & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for stream in 17 18; do
  for view in H0 R0S; do
    [[ -f "${RUN}/stream${stream}/${view}/_SUCCESS" ]] || exit 3
  done
done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

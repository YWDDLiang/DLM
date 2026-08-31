#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-align-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-native-align-eval-%j.out
#SBATCH --error=c3fd-native-align-eval-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GEN_RUN="${ROOT}/runs/c3fd_native_alignment_pool_generation_38857"
readonly COHORT="${ROOT}/cohorts/c3fd_native_alignment_pool_train32x2xK4_v1_20260831"
readonly EVAL_RUN="${ROOT}/runs/c3fd_native_alignment_pool_eval_${SLURM_JOB_ID}"
mkdir "${EVAL_RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${EVAL_RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"
  touch "${EVAL_RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly GENERATION_FINAL_SHA=ff5bb27e2dbdf2c80c1af5ac950c1a8323167ba3bb884729a97218229c1d6e91
readonly GROUPS_SHA=72f2a08a704b211144f7874b9e1f74454082e2548c68e6055a5fdc720a4f792c
readonly ASSEMBLE_REFINED_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly ASSEMBLE_RAW_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly FULL_EVAL_SHA=632c7116faa3738ab1aeadfd7d33d02f21e7b349f41ee874f2a46c753fd200a5
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_METRICS_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71

for path in "${GEN_RUN}/_SUCCESS" "${COHORT}/_SUCCESS" \
  "${V3_SOURCE}/CONFIG.json" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${GEN_RUN}/C3FD_NATIVE_ALIGNMENT_POOL_GENERATION_FINAL.json" | awk '{print $1}')" == "${GENERATION_FINAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/groups.jsonl" | awk '{print $1}')" == "${GROUPS_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/assemble_grounding_repeat.py" | awk '{print $1}')" == "${ASSEMBLE_REFINED_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/assemble_raw_body_repeat.py" | awk '{print $1}')" == "${ASSEMBLE_RAW_SHA}" ]] || exit 2
[[ "$(sha256sum "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" | awk '{print $1}')" == "${FULL_EVAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${V3_SOURCE}/CONFIG.json" | awk '{print $1}')" == "${CONFIG_SHA}" ]] || exit 2
[[ "$(sha256sum "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" | awk '{print $1}')" == "${DIRECT_METRICS_SHA}" ]] || exit 2
[[ "$(sha256sum "${GT_CSV}" | awk '{print $1}')" == "${GT_SHA}" ]] || exit 2
[[ "$(sha256sum "${BASE_RELAX_CACHE}" | awk '{print $1}')" == "${CACHE_SHA}" ]] || exit 2

mkdir -p "${ROOT}/runs/contracts" "${EVAL_RUN}/logs"
{
  printf 'schema=c3fd_native_alignment_pool_eval_v1\n'
  printf 'generation_final=%s\n' "${GENERATION_FINAL_SHA}"
  printf 'groups=%s\n' "${GROUPS_SHA}"
  printf 'policies=82017,82018\n'
  printf 'groups_per_policy=64\n'
  printf 'K=4\n'
  printf 'denominator=256\n'
  printf 'order=raw_then_refined\n'
  printf 'missing=preserve\n'
  printf 'query=false\n'
} >"${EVAL_RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${EVAL_RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir "${CONTRACT_DIR}"
cp "${EVAL_RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_alignment_pool_eval_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'generation_job\t38857\n'
  printf 'policies\t82017,82018\n'
  printf 'groups_per_policy\t64\n'
  printf 'candidates_per_group\t4\n'
  printf 'denominator\t256\n'
  printf 'order\traw_then_refined\n'
  printf 'missing_policy\tpreserve_all_attempts_by_sample_idx\n'
  printf 'policy_selection\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${EVAL_RUN}/scientific_contract.tsv"
sha256sum "${GEN_RUN}/C3FD_NATIVE_ALIGNMENT_POOL_GENERATION_FINAL.json" \
  "${COHORT}/groups.jsonl" "${ROOT}/scripts/assemble_grounding_repeat.py" \
  "${ROOT}/scripts/assemble_raw_body_repeat.py" \
  "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${V3_SOURCE}/CONFIG.json" \
  "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" "${BASE_RELAX_CACHE}" >"${EVAL_RUN}/INPUTS.sha256"

assemble_cell() {
  local policy="$1" label="$2" repeat="$3"
  local body="${GEN_RUN}/policy${policy}/body"
  local refine="${GEN_RUN}/policy${policy}/refine"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${refine}" \
    --output-dir "${EVAL_RUN}/policy${policy}/generation" \
    --arm "${label}" --repeat "${repeat}" --dlm-seed 93117 \
    --refiner-seed 103117 --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${body}" --output-dir "${EVAL_RUN}/raw_policy${policy}/generation" \
    --seed 93 --dlm-seed 93117 --denominator 256 \
    --source-arm full_axis --arm-label "${label}"
}
assemble_cell 82017 control 93017
assemble_cell 82018 candidate 93018

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 2 ]] || exit 2

run_cell() {
  local slot="$1" arm="$2" label="$3"
  local device root direct_pid
  device="${GPU_DEVICES[${slot}]}"
  root="${EVAL_RUN}/${arm}"
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/${arm}-direct.out" 2>"${EVAL_RUN}/logs/${arm}-direct.err" &
  direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/${arm}-full.out" 2>"${EVAL_RUN}/logs/${arm}-full.err"
  wait "${direct_pid}"
}

run_two() {
  local prefix="$1"
  run_cell 0 "${prefix}policy82017" control & p0=$!
  run_cell 1 "${prefix}policy82018" candidate & p1=$!
  wait "${p0}"
  wait "${p1}"
}
run_two "raw_"
run_two ""

for arm in policy82017 policy82018 raw_policy82017 raw_policy82018; do
  [[ -f "${EVAL_RUN}/${arm}/evaluation/direct/report.json" ]] || exit 3
  [[ -f "${EVAL_RUN}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
done
find "${EVAL_RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${EVAL_RUN}/OUTPUTS.sha256"
trap - ERR
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

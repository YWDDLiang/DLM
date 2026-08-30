#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-sft-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-native-sft-eval-%j.out
#SBATCH --error=c3fd-native-sft-eval-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GEN_RUN="${ROOT}/runs/c3fd_native_sft_canary_generation_38745"
readonly COHORT="${ROOT}/cohorts/c3fd_native_sft_canary_train128_val128_v1_20260831"
readonly EVAL_RUN="${ROOT}/runs/c3fd_native_sft_canary_eval_${SLURM_JOB_ID}"
mkdir "${EVAL_RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" \
    >"${EVAL_RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"
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
readonly GENERATION_FINAL_SHA=d25baacbab66238a3380632674093644b71eb7195a7690605b0cd652fb92ce5f
readonly LEDGER_SHA=3a6cae692af6f524e810f8597df28e5024a0cea06be3ecc48c92fb0eb6fb9816
readonly ASSEMBLE_REFINED_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly ASSEMBLE_RAW_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly FULL_EVAL_SHA=632c7116faa3738ab1aeadfd7d33d02f21e7b349f41ee874f2a46c753fd200a5
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_METRICS_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71

for path in "${GEN_RUN}/_SUCCESS" "${COHORT}/_SUCCESS" \
  "${V3_SOURCE}/CONFIG.json" \
  "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${GEN_RUN}/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL.json" | awk '{print $1}')" == "${GENERATION_FINAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/ledger.jsonl" | awk '{print $1}')" == "${LEDGER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/assemble_grounding_repeat.py" | awk '{print $1}')" == "${ASSEMBLE_REFINED_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/assemble_raw_body_repeat.py" | awk '{print $1}')" == "${ASSEMBLE_RAW_SHA}" ]] || exit 2
[[ "$(sha256sum "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" | awk '{print $1}')" == "${FULL_EVAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${V3_SOURCE}/CONFIG.json" | awk '{print $1}')" == "${CONFIG_SHA}" ]] || exit 2
[[ "$(sha256sum "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" | awk '{print $1}')" == "${DIRECT_METRICS_SHA}" ]] || exit 2
[[ "$(sha256sum "${GT_CSV}" | awk '{print $1}')" == "${GT_SHA}" ]] || exit 2
[[ "$(sha256sum "${BASE_RELAX_CACHE}" | awk '{print $1}')" == "${CACHE_SHA}" ]] || exit 2

mkdir -p "${ROOT}/runs/contracts" "${EVAL_RUN}/logs"
{
  printf 'schema=c3fd_native_sft_canary_eval_v1\n'
  printf 'generation_final=%s\n' "${GENERATION_FINAL_SHA}"
  printf 'ledger=%s\n' "${LEDGER_SHA}"
  printf 'assemble_refined=%s\n' "${ASSEMBLE_REFINED_SHA}"
  printf 'assemble_raw=%s\n' "${ASSEMBLE_RAW_SHA}"
  printf 'full_eval=%s\n' "${FULL_EVAL_SHA}"
  printf 'config=%s\n' "${CONFIG_SHA}"
  printf 'direct_metrics=%s\n' "${DIRECT_METRICS_SHA}"
  printf 'gt=%s\n' "${GT_SHA}"
  printf 'cache=%s\n' "${CACHE_SHA}"
  printf 'policies=82017,82018\n'
  printf 'streams=17,18\n'
  printf 'denominator=256\n'
  printf 'order=raw_then_refined\n'
  printf 'missing=preserve_by_sample_idx\n'
  printf 'evaluator_labels=82017:control,82018:candidate_labels_only\n'
  printf 'query=false\n'
} >"${EVAL_RUN}/contract_payload.txt"
readonly CONTRACT_SHA="$(sha256sum "${EVAL_RUN}/contract_payload.txt" | awk '{print $1}')"
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
mkdir "${CONTRACT_DIR}"
cp "${EVAL_RUN}/contract_payload.txt" "${CONTRACT_DIR}/contract_payload.txt"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_native_sft_canary_eval_v1\n'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'generation_job\t38745\n'
  printf 'policies\t82017,82018\n'
  printf 'streams\t17,18\n'
  printf 'source_split\ttrain128,val128_by_frozen_ledger\n'
  printf 'denominator\t256\n'
  printf 'order\traw_then_refined\n'
  printf 'missing_policy\tpreserve_by_sample_idx\n'
  printf 'policy_or_checkpoint_selection\tfalse\n'
  printf 'official_query\tfalse\n'
} >"${EVAL_RUN}/scientific_contract.tsv"

sha256sum "${GEN_RUN}/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL.json" \
  "${COHORT}/ledger.jsonl" "${ROOT}/scripts/assemble_grounding_repeat.py" \
  "${ROOT}/scripts/assemble_raw_body_repeat.py" \
  "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
  "${V3_SOURCE}/CONFIG.json" \
  "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" "${BASE_RELAX_CACHE}" >"${EVAL_RUN}/INPUTS.sha256"

assemble_cell() {
  local stream="$1" policy="$2" label="$3" repeat="$4" dlm_seed="$5" refiner_seed="$6"
  local body="${GEN_RUN}/stream${stream}/policy${policy}/body"
  local refine="${GEN_RUN}/stream${stream}/policy${policy}/refine"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${refine}" \
    --output-dir "${EVAL_RUN}/stream${stream}/policy${policy}/generation" \
    --arm "${label}" --repeat "${repeat}" --dlm-seed "${dlm_seed}" \
    --refiner-seed "${refiner_seed}" --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${body}" \
    --output-dir "${EVAL_RUN}/stream${stream}/raw_policy${policy}/generation" \
    --seed "${stream}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm full_axis --arm-label "${label}"
}

for stream in 17 18; do
  if [[ "${stream}" == 17 ]]; then
    dlm_seed=91117
    refiner_seed=101117
  else
    dlm_seed=92117
    refiner_seed=102117
  fi
  assemble_cell "${stream}" 82017 control $((stream * 100 + 17)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${stream}" 82018 candidate $((stream * 100 + 18)) "${dlm_seed}" "${refiner_seed}"
done

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 4 ]] || exit 2

run_cell() {
  local slot="$1" stream="$2" arm="$3" label="$4"
  local device root direct_pid
  device="${GPU_DEVICES[${slot}]}"
  root="${EVAL_RUN}/stream${stream}/${arm}"
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-direct.out" \
    2>"${EVAL_RUN}/logs/s${stream}-${arm}-direct.err" &
  direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 \
    --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-full.out" \
    2>"${EVAL_RUN}/logs/s${stream}-${arm}-full.err"
  wait "${direct_pid}"
}

run_four() {
  local prefix="$1"
  run_cell 0 17 "${prefix}policy82017" control & p0=$!
  run_cell 1 17 "${prefix}policy82018" candidate & p1=$!
  run_cell 2 18 "${prefix}policy82017" control & p2=$!
  run_cell 3 18 "${prefix}policy82018" candidate & p3=$!
  wait "${p0}"
  wait "${p1}"
  wait "${p2}"
  wait "${p3}"
}

run_four "raw_"
run_four ""

for stream in 17 18; do
  for arm in policy82017 policy82018 raw_policy82017 raw_policy82018; do
    [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
find "${EVAL_RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${EVAL_RUN}/OUTPUTS.sha256"
trap - ERR
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

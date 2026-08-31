#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-llama-eval-%j.out
#SBATCH --error=c3fd-llama-eval-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GEN_RUN="${C3FD_LLAMA_DLM_RUN:?export completed DLM generation run}"
readonly EVAL_RUN="${ROOT}/runs/c3fd_llama_prospective_eval_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly ASSEMBLE_REFINED_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly ASSEMBLE_RAW_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly FULL_EVAL_SHA=632c7116faa3738ab1aeadfd7d33d02f21e7b349f41ee874f2a46c753fd200a5
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_METRICS_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71

mkdir "${EVAL_RUN}"
mkdir "${EVAL_RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${EVAL_RUN}/ENGINEERING_FAILURE.tsv"
  touch "${EVAL_RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${GEN_RUN}/_SUCCESS" ]]
for spec in \
  "${ASSEMBLE_REFINED_SHA}:${ROOT}/scripts/assemble_grounding_repeat.py" \
  "${ASSEMBLE_RAW_SHA}:${ROOT}/scripts/assemble_raw_body_repeat.py" \
  "${FULL_EVAL_SHA}:${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
  "${CONFIG_SHA}:${V3_SOURCE}/CONFIG.json" \
  "${DIRECT_METRICS_SHA}:${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_SHA}:${GT_CSV}" "${CACHE_SHA}:${BASE_RELAX_CACHE}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done

cat >"${EVAL_RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_prospective_eval_v1
routes	F,M
streams	17,18
denominator	256
order	raw_then_refined
planner_and_DLM_failures	preserve_by_sample_idx
evaluator_labels	F:control,M:candidate_labels_only
selection	false
official_query	false
EOF

assemble_cell() {
  local stream="$1" route="$2" label="$3"
  local body="${GEN_RUN}/stream${stream}/${route}/body"
  local refine="${GEN_RUN}/stream${stream}/${route}/refine"
  local dlm_seed refiner_seed
  if [[ "${stream}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117; else dlm_seed=92117; refiner_seed=102117; fi
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${refine}" \
    --output-dir "${EVAL_RUN}/stream${stream}/${route}/generation" \
    --arm "${label}" --repeat "${stream}" --dlm-seed "${dlm_seed}" \
    --refiner-seed "${refiner_seed}" --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${body}" \
    --output-dir "${EVAL_RUN}/stream${stream}/raw_${route}/generation" \
    --seed "${stream}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm full_axis --arm-label "${label}"
}

for stream in 17 18; do
  assemble_cell "${stream}" F control
  assemble_cell "${stream}" M candidate
done

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

run_cell() {
  local slot="$1" stream="$2" arm="$3" label="$4"
  local root="${EVAL_RUN}/stream${stream}/${arm}"
  local device="${allocated_gpus[${slot}]}" direct_pid
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s${stream}-${arm}-direct.err" &
  direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-full.out" 2>"${EVAL_RUN}/logs/s${stream}-${arm}-full.err"
  wait "${direct_pid}"
}

run_four() {
  local prefix="$1"
  run_cell 0 17 "${prefix}F" control & p0=$!
  run_cell 1 17 "${prefix}M" candidate & p1=$!
  run_cell 2 18 "${prefix}F" control & p2=$!
  run_cell 3 18 "${prefix}M" candidate & p3=$!
  wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
}

run_four raw_
run_four ""

for stream in 17 18; do for arm in F M raw_F raw_M; do
  [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/direct/report.json" ]]
  [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]]
done; done
find "${EVAL_RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${EVAL_RUN}/OUTPUTS.sha256"
trap - ERR
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

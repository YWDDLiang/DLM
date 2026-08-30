#!/usr/bin/env bash
#SBATCH --job-name=h1a2-d3po-late-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-d3po-late-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
readonly BODY_RUN="${H1A2_D3PO_LATE_GUIDANCE_RUN:?export H1A2_D3PO_LATE_GUIDANCE_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/d3po_late_guidance_eval_${SLURM_JOB_ID}"
for path in \
  "${BODY_RUN}/_SUCCESS" "${BODY_RUN}/scientific_contract.tsv" \
  "${V3_SOURCE}/CONFIG.json" \
  "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${EVAL_RUN}/logs"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" \
    >"${EVAL_RUN}/ENGINEERING_FAILURE.tsv"
  touch "${EVAL_RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
cp --reflink=auto "${BODY_RUN}/scientific_contract.tsv" \
  "${EVAL_RUN}/generation_scientific_contract.tsv"
sha256sum "${BODY_RUN}/scientific_contract.tsv" \
  "${BODY_RUN}/inputs.sha256" >"${EVAL_RUN}/inputs.sha256"

assemble_cell() {
  local stream="$1" arm="$2" label="$3" repeat="$4" dlm_seed="$5" refiner_seed="$6"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY_RUN}/stream${stream}/${arm}/body" \
    --refine-dir "${BODY_RUN}/stream${stream}/${arm}/refine" \
    --output-dir "${EVAL_RUN}/stream${stream}/${arm}/generation" \
    --arm "${label}" --repeat "${repeat}" \
    --dlm-seed "${dlm_seed}" --refiner-seed "${refiner_seed}" --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${BODY_RUN}/stream${stream}/${arm}/body" \
    --output-dir "${EVAL_RUN}/stream${stream}/raw_${arm}/generation" \
    --seed "${stream}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm full_axis --arm-label "${label}"
}

for stream in 17 18; do
  if [[ "${stream}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117
  else dlm_seed=92117; refiner_seed=102117; fi
  assemble_cell "${stream}" base control $((stream * 10 + 0)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${stream}" d3po_seed81017 candidate $((stream * 10 + 1)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${stream}" d3po_seed81018 candidate $((stream * 10 + 2)) "${dlm_seed}" "${refiner_seed}"
done

run_cell() {
  local gpu="$1" stream="$2" arm="$3" label="$4"
  local root="${EVAL_RUN}/stream${stream}/${arm}"
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
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
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

run_six() {
  local prefix="$1"
  run_cell 0 17 "${prefix}base" control & p0=$!
  run_cell 1 17 "${prefix}d3po_seed81017" candidate & p1=$!
  run_cell 2 17 "${prefix}d3po_seed81018" candidate & p2=$!
  run_cell 3 18 "${prefix}base" control & p3=$!
  run_cell 4 18 "${prefix}d3po_seed81017" candidate & p4=$!
  run_cell 5 18 "${prefix}d3po_seed81018" candidate & p5=$!
  wait "${p0}"; wait "${p1}"; wait "${p2}"
  wait "${p3}"; wait "${p4}"; wait "${p5}"
}

run_six ""
run_six "raw_"

for stream in 17 18; do
  for arm in \
    base d3po_seed81017 d3po_seed81018 \
    raw_base raw_d3po_seed81017 raw_d3po_seed81018; do
    [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
find "${EVAL_RUN}" -type f ! -name 'OUTPUTS.sha256' -print0 \
  | sort -z | xargs -0 sha256sum >"${EVAL_RUN}/OUTPUTS.sha256"
trap - ERR
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

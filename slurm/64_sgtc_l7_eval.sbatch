#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sgtc-l7-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:3
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-sgtc-l7-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
readonly BODY_RUN="${H1A2_SGTC_L7_RUN:?export H1A2_SGTC_L7_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/sgtc_l7_eval_${SLURM_JOB_ID}"
for path in "${BODY_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${EVAL_RUN}/logs"

assemble_cell() {
  local arm="$1" label="$2" repeat="$3"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY_RUN}/seed18/${arm}/body" \
    --refine-dir "${BODY_RUN}/seed18/${arm}/refine" \
    --output-dir "${EVAL_RUN}/seed18/${arm}/generation" \
    --arm "${label}" --repeat "${repeat}" \
    --dlm-seed 92117 --refiner-seed 102117 --denominator 1000
}

assemble_cell base control 180
assemble_cell g0_all candidate 181
assemble_cell g1_strict candidate 182

run_cell() {
  local gpu="$1" arm="$2" label="$3"
  local root="${EVAL_RUN}/seed18/${arm}"
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" \
    --eval-model-name mp20 >"${EVAL_RUN}/logs/s18-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s18-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=1000 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s18-${arm}-full.out" 2>"${EVAL_RUN}/logs/s18-${arm}-full.err"
  wait "${direct_pid}"
}

run_cell 0 base control & p0=$!
run_cell 1 g0_all candidate & p1=$!
run_cell 2 g1_strict candidate & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"

for arm in base g0_all g1_strict; do
  [[ -f "${EVAL_RUN}/seed18/${arm}/evaluation/direct/report.json" ]] || exit 3
  [[ -f "${EVAL_RUN}/seed18/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
done
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

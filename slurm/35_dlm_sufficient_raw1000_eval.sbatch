#!/usr/bin/env bash
#SBATCH --job-name=h1a2-dlm-suff-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-dlm-suff-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GENERATION_RUN="${H1A2_GENERATION_RUN:?export H1A2_GENERATION_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/dlm_sufficient_eval_${SLURM_JOB_ID}"
readonly DLM_SEED=17117
readonly REFINER_SEED=27117

for path in "${GENERATION_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${EVAL_RUNTIME}/collect_official_inputs.py" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

"${PYTHON}" "${ROOT}/scripts/assemble_grounding_two500.py" \
  --round1-body-dir "${GENERATION_RUN}/epoch2/round1/body" \
  --round1-refine-dir "${GENERATION_RUN}/epoch2/round1/refine" \
  --round2-body-dir "${GENERATION_RUN}/epoch2/round2/body" \
  --round2-refine-dir "${GENERATION_RUN}/epoch2/round2/refine" \
  --output-dir "${EVAL_RUN}/control/generation" \
  --arm control --epoch 2 --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}"

"${PYTHON}" "${ROOT}/scripts/assemble_grounding_two500.py" \
  --round1-body-dir "${GENERATION_RUN}/epoch3/round1/body" \
  --round1-refine-dir "${GENERATION_RUN}/epoch3/round1/refine" \
  --round2-body-dir "${GENERATION_RUN}/epoch3/round2/body" \
  --round2-refine-dir "${GENERATION_RUN}/epoch3/round2/refine" \
  --output-dir "${EVAL_RUN}/candidate/generation" \
  --arm candidate --epoch 3 --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}"

run_cell() {
  local arm="$1" gpu="$2"
  local cell_root="${EVAL_RUN}/${arm}"
  mkdir -p "${cell_root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${cell_root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell_root}/evaluation/direct" \
    --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/${arm}-direct.out" 2>"${EVAL_RUN}/logs/${arm}-direct.err" &
  local direct_pid="$!"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${cell_root}/generation/generation_round1.jsonl" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell_root}/evaluation/direct_round1" \
    --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/${arm}-direct-round1.out" 2>"${EVAL_RUN}/logs/${arm}-direct-round1.err" &
  local direct_round1_pid="$!"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${cell_root}/generation/generation_round2.jsonl" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell_root}/evaluation/direct_round2" \
    --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/${arm}-direct-round2.out" 2>"${EVAL_RUN}/logs/${arm}-direct-round2.err" &
  local direct_round2_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=1000 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${arm}" --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/${arm}-full.out" 2>"${EVAL_RUN}/logs/${arm}-full.err"
  wait "${direct_pid}"
  wait "${direct_round1_pid}"
  wait "${direct_round2_pid}"
}

run_cell control 0 & p0=$!
run_cell candidate 1 & p1=$!
wait "${p0}"
wait "${p1}"

env H1_ACTIVE_DENOMINATOR=1000 PYTHONPATH="${EVAL_RUNTIME}" \
  "${PYTHON}" "${EVAL_RUNTIME}/collect_official_inputs.py" \
  --run-root "${EVAL_RUN}" \
  --cell "epoch2=${EVAL_RUN}/control/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl" \
  --cell "epoch3=${EVAL_RUN}/candidate/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl" \
  >"${EVAL_RUN}/logs/collect-official-inputs.out" 2>"${EVAL_RUN}/logs/collect-official-inputs.err"

touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

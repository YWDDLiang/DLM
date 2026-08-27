#!/usr/bin/env bash
#SBATCH --job-name=h1a2-energy-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-energy-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY_RUN="${H1A2_ENERGY_PAIR_BODY_RUN:?export H1A2_ENERGY_PAIR_BODY_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/dlm_energy_pair_eval_${SLURM_JOB_ID}"
readonly DLM_SEEDS=(31017 32017 33017 34017)
readonly REFINER_SEEDS=(41017 42017 43017 44017)
readonly ARMS=(control candidate control candidate)

for path in "${BODY_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

for stream in 0 1 2 3; do
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY_RUN}/stream${stream}/body" \
    --refine-dir "${BODY_RUN}/stream${stream}/refine" \
    --output-dir "${EVAL_RUN}/stream${stream}/generation" \
    --arm "${ARMS[$stream]}" --repeat "${stream}" \
    --dlm-seed "${DLM_SEEDS[$stream]}" --refiner-seed "${REFINER_SEEDS[$stream]}" \
    --denominator 256
done

run_stream() {
  local stream="$1" gpu="$2" arm="${ARMS[$1]}"
  local cell_root="${EVAL_RUN}/stream${stream}"
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
    >"${EVAL_RUN}/logs/stream${stream}-direct.out" 2>"${EVAL_RUN}/logs/stream${stream}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${arm}" --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/stream${stream}-full.out" 2>"${EVAL_RUN}/logs/stream${stream}-full.err"
  wait "${direct_pid}"
}

run_stream 0 0 & p0=$!
run_stream 1 1 & p1=$!
run_stream 2 2 & p2=$!
run_stream 3 3 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for stream in 0 1 2 3; do
  [[ -f "${EVAL_RUN}/stream${stream}/evaluation/direct/report.json" ]] || exit 3
  [[ -f "${EVAL_RUN}/stream${stream}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
done
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

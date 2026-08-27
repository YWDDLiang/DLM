#!/usr/bin/env bash
#SBATCH --job-name=h1a2-atom-major-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-atom-major-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY_RUN="${H1A2_ATOM_MAJOR_RUN:?export H1A2_ATOM_MAJOR_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/dlm_atom_major_l6_eval_${SLURM_JOB_ID}"

for path in "${BODY_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

for seed in 17 18; do
  if [[ "${seed}" == 17 ]]; then dlm_seed=17117; refiner_seed=27117; else dlm_seed=18117; refiner_seed=28117; fi
  for arm in full_atom hard_atom; do
    "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
      --body-dir "${BODY_RUN}/seed${seed}/${arm}/body" \
      --refine-dir "${BODY_RUN}/seed${seed}/${arm}/refine" \
      --output-dir "${EVAL_RUN}/seed${seed}/${arm}/generation" \
      --arm candidate --repeat $((seed * 10 + 4)) \
      --dlm-seed "${dlm_seed}" --refiner-seed "${refiner_seed}" \
      --denominator 256 --refinement-steps 800
  done
done

run_cell() {
  local gpu="$1" seed="$2" arm="$3"
  local cell_root="${EVAL_RUN}/seed${seed}/${arm}"
  mkdir -p "${cell_root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${cell_root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell_root}/evaluation/direct" --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/s${seed}-${arm}-direct.out" \
    2>"${EVAL_RUN}/logs/s${seed}-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm candidate --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${seed}-${arm}-full.out" \
    2>"${EVAL_RUN}/logs/s${seed}-${arm}-full.err"
  wait "${direct_pid}"
}

run_cell 0 17 full_atom & p0=$!
run_cell 1 17 hard_atom & p1=$!
run_cell 2 18 full_atom & p2=$!
run_cell 3 18 hard_atom & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

for seed in 17 18; do
  for arm in full_atom hard_atom; do
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

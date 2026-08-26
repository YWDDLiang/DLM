#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-ckpt-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-ground-ckpt-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SWEEP_RUN="${H1A2_SWEEP_RUN:?export H1A2_SWEEP_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/grounding_checkpoint_eval_${SLURM_JOB_ID}"
readonly STEPS=(500 1000)
readonly DLM_SEED=17117
readonly REFINER_SEED=27117

for path in "${SWEEP_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${EVAL_RUNTIME}/collect_official_inputs.py" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

for step in "${STEPS[@]}"; do
  label="$(printf '%04d' "${step}")"
  for arm in control candidate; do
    cell="step${label}/${arm}"
    "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
      --body-dir "${SWEEP_RUN}/${cell}/body" \
      --refine-dir "${SWEEP_RUN}/${cell}/refine" \
      --output-dir "${EVAL_RUN}/${cell}/generation" \
      --arm "${arm}" --repeat "${step}" \
      --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" \
      --denominator 256
  done
done

run_cell() {
  local step="$1" arm="$2" gpu="$3"
  local label cell cell_root
  label="$(printf '%04d' "${step}")"
  cell="step${label}/${arm}"
  cell_root="${EVAL_RUN}/${cell}"
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
    >"${EVAL_RUN}/logs/s${step}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s${step}-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${arm}" --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${step}-${arm}-full.out" 2>"${EVAL_RUN}/logs/s${step}-${arm}-full.err"
  wait "${direct_pid}"
}

run_cell 500 control 0 & p0=$!
run_cell 500 candidate 1 & p1=$!
run_cell 1000 control 2 & p2=$!
run_cell 1000 candidate 3 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

cells=()
for step in "${STEPS[@]}"; do
  label="$(printf '%04d' "${step}")"
  for arm in control candidate; do
    labels="${EVAL_RUN}/step${label}/${arm}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
    cells+=(--cell "s${step}_${arm}=${labels}")
  done
done
env H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}" \
  "${PYTHON}" "${EVAL_RUNTIME}/collect_official_inputs.py" \
  --run-root "${EVAL_RUN}" "${cells[@]}" \
  >"${EVAL_RUN}/logs/collect-official-inputs.out" 2>"${EVAL_RUN}/logs/collect-official-inputs.err"

touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

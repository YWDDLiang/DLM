#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ground-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-ground-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly REPEAT_RUN="${H1A2_REPEAT_RUN:?export H1A2_REPEAT_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUN="${ROOT}/runs/grounding_repeat4_eval_${SLURM_JOB_ID}"
readonly DLM_SEEDS=(17117 17217 17317 17417)
readonly REFINER_SEEDS=(27117 27217 27317 27417)

for path in "${REPEAT_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

for idx in 0 1 2 3; do
  repeat="$((idx + 1))"
  for arm in control candidate; do
    cell="repeat$(printf '%02d' "${repeat}")/${arm}"
    "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
      --body-dir "${REPEAT_RUN}/${cell}/body" \
      --refine-dir "${REPEAT_RUN}/${cell}/refine" \
      --output-dir "${EVAL_RUN}/${cell}/generation" \
      --arm "${arm}" --repeat "${repeat}" \
      --dlm-seed "${DLM_SEEDS[$idx]}" --refiner-seed "${REFINER_SEEDS[$idx]}" \
      --denominator 256
  done
done

run_cell() {
  local repeat="$1" arm="$2" gpu="$3"
  local cell="repeat$(printf '%02d' "${repeat}")/${arm}"
  local cell_root="${EVAL_RUN}/${cell}"
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
    >"${EVAL_RUN}/logs/r${repeat}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/r${repeat}-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${V3_SOURCE}:${R03E}/runtime" \
    "${PYTHON}" "${V3_SOURCE}/run_full_reconstructed_eval.py" \
    --arm "${arm}" --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/r${repeat}-${arm}-full.out" 2>"${EVAL_RUN}/logs/r${repeat}-${arm}-full.err"
  wait "${direct_pid}"
}

pids=()
gpu=0
for repeat in 1 2 3 4; do
  for arm in control candidate; do
    run_cell "${repeat}" "${arm}" "${gpu}" &
    pids+=("$!")
    gpu="$((gpu + 1))"
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

cells=()
for repeat in 1 2 3 4; do
  for arm in control candidate; do
    cell_id="r${repeat}_${arm}"
    labels="${EVAL_RUN}/repeat$(printf '%02d' "${repeat}")/${arm}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
    cells+=(--cell "${cell_id}=${labels}")
  done
done
env H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${V3_SOURCE}" \
  "${PYTHON}" "${V3_SOURCE}/collect_official_inputs.py" \
  --run-root "${EVAL_RUN}" "${cells[@]}" \
  >"${EVAL_RUN}/logs/collect-official-inputs.out" 2>"${EVAL_RUN}/logs/collect-official-inputs.err"

touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

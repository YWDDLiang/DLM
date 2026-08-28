#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-labels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-ctv-labels-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SPLIT="${H1A2_CTV_BRANCH_SPLIT:?set train or validation}"
readonly ROLLOUT="${H1A2_CTV_ROLLOUT_RUN:?export H1A2_CTV_ROLLOUT_RUN}"
readonly COHORTS="${H1A2_CTV_CERTIFIED_COHORTS:?export H1A2_CTV_CERTIFIED_COHORTS}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly RUN_DIR="${ROOT}/runs/ctv_branch_labels_${SPLIT}_${SLURM_JOB_ID}"

case "${SPLIT}" in
  train)
    readonly PLANS="${COHORTS}/CTV_BRANCH_TRAIN_PLANS.jsonl"
    readonly EXPECTED=2048
    readonly CHUNKS=8
    ;;
  validation)
    readonly PLANS="${COHORTS}/CTV_BRANCH_VAL_PLANS.jsonl"
    readonly EXPECTED=1024
    readonly CHUNKS=4
    ;;
  *)
    printf 'invalid CTV split: %s\n' "${SPLIT}" >&2
    exit 2
    ;;
esac

for path in \
  "${ROLLOUT}/_SUCCESS" \
  "${COHORTS}/_SUCCESS" \
  "${PLANS}" \
  "${V3_SOURCE}/CONFIG.json" \
  "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
  "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_CSV}" \
  "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { printf 'missing %s\n' "${path}" >&2; exit 2; }
done

mkdir -p "${RUN_DIR}/evaluation" "${RUN_DIR}/logs"
"${PYTHON}" "${ROOT}/scripts/assemble_ctv_branch_generation.py" \
  --rollout-run "${ROLLOUT}" \
  --plans-jsonl "${PLANS}" \
  --output-dir "${RUN_DIR}/generation" \
  --split "${SPLIT}" \
  --expected-branches "${EXPECTED}"

run_chunk() {
  local chunk="$1" gpu="$2" name
  name="$(printf 'chunk%02d' "${chunk}")"
  local generation="${RUN_DIR}/generation/${name}/generation.jsonl"
  local output="${RUN_DIR}/evaluation/${name}"
  mkdir -p "${output}"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${output}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${generation}" \
    --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${output}/direct" \
    --eval-model-name mp20 \
    >"${RUN_DIR}/logs/${name}-direct.out" \
    2>"${RUN_DIR}/logs/${name}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" H1_ACTIVE_DENOMINATOR=256 \
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm candidate --repeat 0 \
    --generation-jsonl "${generation}" \
    --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" \
    --working-relax-cache "${output}/relax_cache.jsonl" \
    --output-dir "${output}/full_reconstructed" \
    >"${RUN_DIR}/logs/${name}-full.out" \
    2>"${RUN_DIR}/logs/${name}-full.err"
  wait "${direct_pid}"
}

for ((wave=0; wave<CHUNKS; wave+=4)); do
  pids=()
  for gpu in 0 1 2 3; do
    chunk=$((wave + gpu))
    if (( chunk < CHUNKS )); then
      run_chunk "${chunk}" "${gpu}" &
      pids+=("$!")
    fi
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
done

for ((chunk=0; chunk<CHUNKS; chunk++)); do
  name="$(printf 'chunk%02d' "${chunk}")"
  [[ -f "${RUN_DIR}/evaluation/${name}/direct/report.json" ]] || exit 3
  [[ -f "${RUN_DIR}/evaluation/${name}/full_reconstructed/_SUCCESS" ]] || exit 3
done

"${PYTHON}" "${ROOT}/scripts/merge_ctv_branch_labels.py" \
  --generation-root "${RUN_DIR}/generation" \
  --evaluation-root "${RUN_DIR}/evaluation" \
  --output-dir "${RUN_DIR}/labels" \
  --expected-branches "${EXPECTED}"
touch "${RUN_DIR}/_SUCCESS"
printf '%s\n' "${RUN_DIR}"

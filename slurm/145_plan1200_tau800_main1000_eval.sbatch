#!/usr/bin/env bash
#SBATCH --job-name=plan1200-tau800-final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=plan1200-tau800-final-%j.out
#SBATCH --error=plan1200-tau800-final-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly INPUT="${ROOT}/runs/plan1200_tau800_main1000_repartition_20260902_v1"
readonly RUN="${ROOT}/runs/plan1200_tau800_main1000_final_eval_${SLURM_JOB_ID}"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime_plan1200_20260902_v1"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${INPUT}/_SUCCESS" ]]
[[ "$(wc -l < "${INPUT}/generation.jsonl")" -eq 1000 ]]
[[ -f "${INPUT}/relax_cache.jsonl" ]]

env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 H1_ACTIVE_DENOMINATOR=1000 \
  PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
  "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
  --arm candidate --repeat 0 --generation-jsonl "${INPUT}/generation.jsonl" \
  --config "${SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
  --working-relax-cache "${INPUT}/relax_cache.jsonl" \
  --output-dir "${RUN}/full_reconstructed"

[[ -f "${RUN}/full_reconstructed/_SUCCESS" ]]
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"

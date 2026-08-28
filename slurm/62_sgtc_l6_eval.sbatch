#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sgtc-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-sgtc-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
readonly BODY_RUN="${H1A2_SGTC_L6_RUN:?export H1A2_SGTC_L6_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/sgtc_l6_eval_${SLURM_JOB_ID}"
for path in "${BODY_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${EVAL_RUN}/logs"

assemble_cell() {
  local seed="$1" arm="$2" label="$3" repeat="$4" dlm_seed="$5" refiner_seed="$6"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY_RUN}/seed${seed}/${arm}/body" \
    --refine-dir "${BODY_RUN}/seed${seed}/${arm}/refine" \
    --output-dir "${EVAL_RUN}/seed${seed}/${arm}/generation" \
    --arm "${label}" --repeat "${repeat}" \
    --dlm-seed "${dlm_seed}" --refiner-seed "${refiner_seed}" --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${BODY_RUN}/seed${seed}/${arm}/body" \
    --output-dir "${EVAL_RUN}/seed${seed}/raw_${arm}/generation" \
    --seed "${seed}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm "${arm}" --arm-label "${label}"
}

for seed in 17 18; do
  if [[ "${seed}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117; else dlm_seed=92117; refiner_seed=102117; fi
  assemble_cell "${seed}" base control $((seed * 10 + 0)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${seed}" g0_all candidate $((seed * 10 + 1)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${seed}" g1_strict candidate $((seed * 10 + 2)) "${dlm_seed}" "${refiner_seed}"
done

run_cell() {
  local gpu="$1" seed="$2" arm="$3" label="$4"
  local root="${EVAL_RUN}/seed${seed}/${arm}"
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" \
    --eval-model-name mp20 >"${EVAL_RUN}/logs/s${seed}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s${seed}-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${seed}-${arm}-full.out" 2>"${EVAL_RUN}/logs/s${seed}-${arm}-full.err"
  wait "${direct_pid}"
}

run_cell 0 17 base control & p0=$!
run_cell 1 17 g0_all candidate & p1=$!
run_cell 2 17 g1_strict candidate & p2=$!
run_cell 3 18 base control & p3=$!
run_cell 4 18 g0_all candidate & p4=$!
run_cell 5 18 g1_strict candidate & p5=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"; wait "${p4}"; wait "${p5}"

run_cell 0 17 raw_base control & q0=$!
run_cell 1 17 raw_g0_all candidate & q1=$!
run_cell 2 17 raw_g1_strict candidate & q2=$!
run_cell 3 18 raw_base control & q3=$!
run_cell 4 18 raw_g0_all candidate & q4=$!
run_cell 5 18 raw_g1_strict candidate & q5=$!
wait "${q0}"; wait "${q1}"; wait "${q2}"; wait "${q3}"; wait "${q4}"; wait "${q5}"

for seed in 17 18; do
  for arm in base g0_all g1_strict raw_base raw_g0_all raw_g1_strict; do
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

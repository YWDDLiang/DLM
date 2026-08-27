#!/usr/bin/env bash
#SBATCH --job-name=h1a2-cond-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-cond-eval-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY_RUN="${H1A2_CONDITION_SCHEDULE_RUN:?export H1A2_CONDITION_SCHEDULE_RUN}"
readonly V3_SOURCE="${H1A2_V3_SOURCE:?export H1A2_V3_SOURCE}"
readonly V3_RUN="${H1A2_V3_RUN:?export H1A2_V3_RUN}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly EVAL_RUN="${ROOT}/runs/dlm_condition_schedule_l6_eval_${SLURM_JOB_ID}"

for path in "${BODY_RUN}/_SUCCESS" "${V3_SOURCE}/CONFIG.json" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" "${GT_CSV}" "${BASE_RELAX_CACHE}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${EVAL_RUN}/logs"

assemble_cell() {
  local seed="$1" arm="$2" arm_label="$3" repeat="$4" dlm_seed="$5" refiner_seed="$6"
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY_RUN}/seed${seed}/${arm}/body" \
    --refine-dir "${BODY_RUN}/seed${seed}/${arm}/refine" \
    --output-dir "${EVAL_RUN}/seed${seed}/${arm}/generation" \
    --arm "${arm_label}" --repeat "${repeat}" \
    --dlm-seed "${dlm_seed}" --refiner-seed "${refiner_seed}" \
    --denominator 256
}

for seed in 17 18; do
  if [[ "${seed}" == 17 ]]; then dlm_seed=17117; refiner_seed=27117; else dlm_seed=18117; refiner_seed=28117; fi
  assemble_cell "${seed}" full_axis control $((seed * 10 + 0)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${seed}" full_joint candidate $((seed * 10 + 1)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${seed}" hard_axis candidate $((seed * 10 + 2)) "${dlm_seed}" "${refiner_seed}"
  assemble_cell "${seed}" hard_joint candidate $((seed * 10 + 3)) "${dlm_seed}" "${refiner_seed}"
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${BODY_RUN}/seed${seed}/full_axis/body" \
    --output-dir "${EVAL_RUN}/seed${seed}/raw_full_axis/generation" \
    --seed "${seed}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm full_axis --arm-label control
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${BODY_RUN}/seed${seed}/hard_joint/body" \
    --output-dir "${EVAL_RUN}/seed${seed}/raw_hard_joint/generation" \
    --seed "${seed}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm hard_joint --arm-label candidate
done

run_cell() {
  local gpu="$1" seed="$2" arm="$3" arm_label="$4"
  local cell_root="${EVAL_RUN}/seed${seed}/${arm}"
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
    >"${EVAL_RUN}/logs/s${seed}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s${seed}-${arm}-direct.err" &
  local direct_pid="$!"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${arm_label}" --repeat 0 \
    --generation-jsonl "${cell_root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell_root}/relax_cache.jsonl" \
    --output-dir "${cell_root}/evaluation/full_reconstructed" \
    >"${EVAL_RUN}/logs/s${seed}-${arm}-full.out" 2>"${EVAL_RUN}/logs/s${seed}-${arm}-full.err"
  wait "${direct_pid}"
}

run_cell 0 17 full_axis control & p0=$!
run_cell 1 17 full_joint candidate & p1=$!
run_cell 2 17 hard_axis candidate & p2=$!
run_cell 3 17 hard_joint candidate & p3=$!
run_cell 4 18 full_axis control & p4=$!
run_cell 5 18 full_joint candidate & p5=$!
run_cell 6 18 hard_axis candidate & p6=$!
run_cell 7 18 hard_joint candidate & p7=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
wait "${p4}"; wait "${p5}"; wait "${p6}"; wait "${p7}"

# Diagnostic only: isolate the effect of model494 by applying the same common
# evaluator/CHGNet relaxation directly to the unrefined full-axis DLM bodies.
run_cell 0 17 raw_full_axis control & p8=$!
run_cell 2 17 raw_hard_joint candidate & p9=$!
run_cell 4 18 raw_full_axis control & p10=$!
run_cell 6 18 raw_hard_joint candidate & p11=$!
wait "${p8}"; wait "${p9}"; wait "${p10}"; wait "${p11}"

for seed in 17 18; do
  for arm in full_axis full_joint hard_axis hard_joint; do
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
for seed in 17 18; do
  for arm in raw_full_axis raw_hard_joint; do
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/direct/report.json" ]] || exit 3
    [[ -f "${EVAL_RUN}/seed${seed}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]] || exit 3
  done
done
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"

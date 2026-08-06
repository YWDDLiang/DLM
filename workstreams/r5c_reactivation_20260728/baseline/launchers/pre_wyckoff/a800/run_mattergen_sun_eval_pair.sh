#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/20260524_mattergen_sun_eval}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}/notes" "${RUN_ROOT}/outputs/lowrl5" "${RUN_ROOT}/outputs/sftbest"

run_one() {
  local name="$1"
  local structures_path="$2"
  local output_dir="${RUN_ROOT}/outputs/${name}"
  local notes_prefix="${RUN_ROOT}/notes/${name}"

  "${PYTHON_BIN}" scripts/run_mattergen_sun_eval.py \
    --structures-path "${structures_path}" \
    --reference-dataset "${REFERENCE_DATASET}" \
    --save-as "${notes_prefix}_metrics.json" \
    --save-detailed-as "${notes_prefix}_detailed_metrics.json" \
    --structures-output-path "${output_dir}/relaxed.extxyz" \
    --summary-json "${notes_prefix}_summary.json" \
    --relax-failures-json "${notes_prefix}_relax_failures.json" \
    --unsupported-failures-json "${notes_prefix}_unsupported_failures.json" \
    --metric-errors-json "${notes_prefix}_metric_errors.json" \
    --relax-max-steps "${RELAX_MAX_STEPS:-500}" \
    --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}" \
    --device cuda \
    --structure-matcher disordered

  if [ -f "${notes_prefix}_detailed_metrics.json" ]; then
    python scripts/analyze_mattergen_sun_detailed.py \
      --summary-json "${notes_prefix}_summary.json" \
      --detailed-json "${notes_prefix}_detailed_metrics.json" \
      --label "${name}" \
      --output-json "${notes_prefix}_threshold_analysis.json" \
      --output-md "${notes_prefix}_threshold_analysis.md"
  fi
}

run_one "lowrl5" "${RUN_ROOT}/outputs/lowrl5/generated.extxyz"
run_one "sftbest" "${RUN_ROOT}/outputs/sftbest/generated.extxyz"

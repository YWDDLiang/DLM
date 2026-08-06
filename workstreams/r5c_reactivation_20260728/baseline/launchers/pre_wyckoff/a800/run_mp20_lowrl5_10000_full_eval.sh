#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"

GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-10000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
BASE_MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
SAMPLE_MASTER_PORT="${SAMPLE_MASTER_PORT:-${BASE_MASTER_PORT}}"
REFINE_MASTER_PORT="${REFINE_MASTER_PORT:-$((BASE_MASTER_PORT + 1))}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
SAMPLE_DIR="${RUN_DIR}/outputs/sample10000"
REFINED_DIR="${RUN_DIR}/outputs/refined10000"
SUN_DIR="${RUN_DIR}/outputs/mattergen_sun10000"
NOTES_DIR="${RUN_DIR}/notes"
REPORT_PATH="reports/${RUN_ID}_crysllmgen_sun10000.md"
mkdir -p "${SAMPLE_DIR}" "${REFINED_DIR}" "${SUN_DIR}" "${NOTES_DIR}" reports

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "temperature": float("${TEMPERATURE}"),
    "target_graph_success": int("${TARGET_GRAPH_SUCCESS}"),
    "max_attempts": int("${MAX_ATTEMPTS}"),
    "sample_batch_size": int("${SAMPLE_BATCH_SIZE}"),
    "refine_batch_size": int("${REFINE_BATCH_SIZE}"),
    "diff_steps": int("${DIFF_STEPS}"),
    "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
    "mattergen_root": "${MATTERGEN_ROOT}",
    "reference_dataset": "${REFERENCE_DATASET}",
    "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
    "selection_basis": "lowrl5 had best 1000-sample MatterGen S.U.N among tested candidates",
}, indent=2) + "\n")
PY

sample_args=(
  --checkpoint-path "${CHECKPOINT_PATH}"
  --output-dir "${SAMPLE_DIR}"
  --target-graph-success "${TARGET_GRAPH_SUCCESS}"
  --max-attempts "${MAX_ATTEMPTS}"
  --num-samples "${MAX_ATTEMPTS}"
  --batch-size "${SAMPLE_BATCH_SIZE}"
  --block-length 1
  --temperature "${TEMPERATURE}"
  --generation-schedule "${GENERATION_SCHEDULE}"
  --schema-logit-mask
  --prefill-slot-tokens
  --atom-count-grammar-mask
  --duplicate-coordinate-mask
  --lattice-volume-mask
)

torchrun --nproc_per_node=2 --master_port "${SAMPLE_MASTER_PORT}" \
  scripts/sample_llada_crystals.py "${sample_args[@]}"

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/sample10000_distribution.json" \
  --output-md "${NOTES_DIR}/sample10000_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --refined-world-size "${REFINED_WORLD_SIZE}" \
  --output-json "${NOTES_DIR}/sample10000_composition_raw.json" \
  --output-md "${NOTES_DIR}/sample10000_composition_raw.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample10000_failure_modes_raw.json" \
  --output-md "${NOTES_DIR}/sample10000_failure_modes_raw.md"

torchrun --nproc_per_node=2 --master_port "${REFINE_MASTER_PORT}" \
  scripts/refine_dlm_with_crysllmgen.py \
  --proposal-graphs "${SAMPLE_DIR}/proposal_graphs.pt" \
  --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
  --output-dir "${REFINED_DIR}" \
  --batch-size "${REFINE_BATCH_SIZE}" \
  --diff-steps "${DIFF_STEPS}" \
  --max-proposals "${TARGET_GRAPH_SUCCESS}"

python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED_DIR}" \
  --output-json "${NOTES_DIR}/crysllmgen_metrics10000.json"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --refined-pt "${REFINED_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt" \
  --refined-world-size "${REFINED_WORLD_SIZE}" \
  --output-json "${NOTES_DIR}/composition10000.json" \
  --output-md "${NOTES_DIR}/composition10000.md"

"${MATTERGEN_PYTHON}" scripts/convert_crysllmgen_pt_to_extxyz.py \
  --input-pt "${REFINED_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt" \
  --output-extxyz "${SUN_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/mattergen_sun10000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/mattergen_sun10000_detailed_metrics.json"
  --structures-output-path "${SUN_DIR}/relaxed.extxyz"
  --summary-json "${NOTES_DIR}/mattergen_sun10000_summary.json"
  --relax-failures-json "${NOTES_DIR}/mattergen_sun10000_relax_failures.json"
  --unsupported-failures-json "${NOTES_DIR}/mattergen_sun10000_unsupported_failures.json"
  --metric-errors-json "${NOTES_DIR}/mattergen_sun10000_metric_errors.json"
  --relax-max-steps "${RELAX_MAX_STEPS:-500}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi

"${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

if [ -f "${NOTES_DIR}/mattergen_sun10000_detailed_metrics.json" ]; then
  python scripts/analyze_mattergen_sun_detailed.py \
    --summary-json "${NOTES_DIR}/mattergen_sun10000_summary.json" \
    --detailed-json "${NOTES_DIR}/mattergen_sun10000_detailed_metrics.json" \
    --label "${RUN_ID}" \
    --output-json "${NOTES_DIR}/mattergen_sun10000_threshold_analysis.json" \
    --output-md "${NOTES_DIR}/mattergen_sun10000_threshold_analysis.md"
fi

python scripts/report_mp20_final10000_eval.py \
  --run-id "${RUN_ID}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --generation-schedule "${GENERATION_SCHEDULE}" \
  --temperature "${TEMPERATURE}" \
  --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/composition10000.json" \
  --failure-modes "${NOTES_DIR}/sample10000_failure_modes_raw.json" \
  --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics10000.json" \
  --sun-summary "${NOTES_DIR}/mattergen_sun10000_summary.json" \
  --baseline-crysllmgen-metrics "runs/20260522_142200-lowrl5-nelemseq-refined1000/notes/crysllmgen_metrics1000.json" \
  --baseline-composition-summary "runs/20260522_142200-lowrl5-nelemseq-refined1000/notes/composition1000.json" \
  --baseline-sun-summary "runs/20260524_mattergen_sun_eval/notes/lowrl5_summary.json" \
  --output-md "${REPORT_PATH}"

cp "${REPORT_PATH}" "${NOTES_DIR}/final10000_report.md"
echo "FINAL_REPORT=${PROJECT_ROOT}/${REPORT_PATH}"

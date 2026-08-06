#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260529_r5_ceg_bdlm_prompt_v0}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
R2_CHECKPOINT="${R2_CHECKPOINT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260527_semalign_selfimprove_r2/outputs/stage_b/final}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
R5_DATA_DIR="${R5_DATA_DIR:-data/dlm_sft/mp_20_r5_z_prompt_v0}"
PROTOTYPE_JSONL="${PROTOTYPE_JSONL:-data/prototypes/mp20_stable_prototype_library.jsonl}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
GPU_COUNT="${GPU_COUNT:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-512}"
SFT_LR="${SFT_LR:-8e-7}"
SFT_MAX_TRAIN_STEPS="${SFT_MAX_TRAIN_STEPS:-1000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
RUN_1000_IF_PASS="${RUN_1000_IF_PASS:-1}"
FORCE_1000="${FORCE_1000:-0}"
BASELINE_COMP_VALID_256="${BASELINE_COMP_VALID_256:-0.91796875}"
BASELINE_STRICT_VALID_256="${BASELINE_STRICT_VALID_256:-0.640625}"
BASELINE_HIGH_SYM_256="${BASELINE_HIGH_SYM_256:-0.6526691547831254}"
MASTER_PORT_BASE="${MASTER_PORT:-$((28000 + (${SLURM_JOB_ID:-0} % 20000)))}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [ ! -d "${R2_CHECKPOINT}" ]; then
  echo "R2_CHECKPOINT does not exist: ${R2_CHECKPOINT}" >&2
  exit 2
fi

RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}" "$(dirname "${PROTOTYPE_JSONL}")"

PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((MASTER_PORT_BASE + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "===== COMMAND $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  printf '%q ' "$@" | tee -a "${log_file}"
  echo | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== STATUS ${status} $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  return "${status}"
}

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "route": "R5 CEG-BDLM prompt-side z conditioning v0",
  "model_path": "${MODEL_PATH}",
  "r2_checkpoint": "${R2_CHECKPOINT}",
  "base_data_dir": "${BASE_DATA_DIR}",
  "r5_data_dir": "${R5_DATA_DIR}",
  "prototype_jsonl": "${PROTOTYPE_JSONL}",
  "temperature": float("${TEMPERATURE}"),
  "generation_schedule": "${GENERATION_SCHEDULE}",
  "gpu_count": int("${GPU_COUNT}"),
  "no_10000": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/unit_tests.log" \
  python -m unittest tests.test_r5_conditioning tests.test_llada_generation_masks tests.test_fixed_slot

run_logged "${LOG_DIR}/py_compile.log" \
  python -m py_compile \
    crystal_dlm/r5_conditioning.py \
    scripts/build_r5_z_prompt_sft_data.py \
    scripts/analyze_r5_conditional_buckets.py \
    scripts/sample_llada_crystals.py \
    scripts/llada_sft.py

if [ ! -f "${BASE_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

run_logged "${LOG_DIR}/r5_0_conditional_diagnostics.log" \
  python scripts/analyze_r5_conditional_buckets.py \
    --run "R2=runs/20260527_semalign_selfimprove_r2" \
    --run "ABL1=runs/20260528_stcompress_abl1_refined1000" \
    --run "R3=runs/20260529_r3_e3_physical_header_r2abs_fix2" \
    --run "R4=runs/20260529_r4_e4a_geometry_inpaint_r2abs" \
    --output-json "${NOTES_DIR}/r5_0_conditional_diagnostics.json" \
    --output-md "${NOTES_DIR}/r5_0_conditional_diagnostics.md" || true

if [ "${REBUILD_R5_DATA:-0}" = "1" ]; then
  rm -rf "${R5_DATA_DIR}"
fi
if [ ! -f "${R5_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_r5_z_prompt_data.log" \
    python scripts/build_r5_z_prompt_sft_data.py \
      --input-dir "${BASE_DATA_DIR}" \
      --csv-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${R5_DATA_DIR}" \
      --prototype-jsonl "${PROTOTYPE_JSONL}" \
      --tokenizer-path "${MODEL_PATH}" \
      --replay-fraction 0.25 \
      --formula-weight-cap 64
fi
cp "${R5_DATA_DIR}/r5_z_prompt_summary.json" "${NOTES_DIR}/input_r5_z_prompt_summary.json" || true
cp "${R5_DATA_DIR}/result.md" "${NOTES_DIR}/input_r5_z_prompt_result.md" || true

train_sft() {
  local name="$1"
  local output_dir="$2"
  local limit_args=()
  shift 2
  limit_args=("$@")
  if [ -f "${output_dir}/final/adapter_config.json" ] || [ -f "${output_dir}/final/config.json" ]; then
    echo "Reusing existing SFT checkpoint ${output_dir}/final"
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${name}_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${R2_CHECKPOINT}" \
      --representation fixed_slot \
      --data-dir "${R5_DATA_DIR}" \
      --output-dir "${output_dir}" \
      --max-length "${SFT_MAX_LENGTH}" \
      --epochs 1 \
      --max-train-steps "${SFT_MAX_TRAIN_STEPS}" \
      --batch-size "${SFT_BATCH_SIZE}" \
      --grad-accum "${SFT_GRAD_ACCUM}" \
      --lr "${SFT_LR}" \
      --lr-scheduler cosine \
      --warmup-steps 20 \
      --min-lr-ratio 0.2 \
      --weighted-sampling \
      --atom-count-loss-weight 3.0 \
      --slot-marker-loss-weight 0.25 \
      --empty-slot-loss-weight 0.20 \
      --nonempty-slot-loss-weight 2.5 \
      --late-nonempty-slot-loss-weight 4.0 \
      --coordinate-loss-weight 1.1 \
      --pad-coordinate-loss-weight 0.10 \
      --train-prefill-slot-tokens \
      --logging-steps 20 \
      --eval-steps 250 \
      --save-steps 1000 \
      --eval-max-batches 30 \
      --position-diagnostics-steps 250 \
      ${limit_args[@]+"${limit_args[@]}"}
}

sample_fixed() {
  local name="$1"
  local checkpoint="$2"
  local sample_dir="$3"
  local sample_count="$4"
  shift 4
  local extra_args=("$@")
  if [ -f "${sample_dir}/sample_metrics.json" ]; then
    echo "Reusing existing sample dir ${sample_dir}"
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${name}_sample.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${sample_dir}" \
      --num-samples "${sample_count}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --temperature "${TEMPERATURE}" \
      --block-length 1 \
      --generation-schedule "${GENERATION_SCHEDULE}" \
      --schema-logit-mask \
      --prefill-slot-tokens \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask \
      --prompt-jsonl "${R5_DATA_DIR}/prototype_prompt_pool.jsonl" \
      ${extra_args[@]+"${extra_args[@]}"}
}

analyze_sample() {
  local name="$1"
  local sample_dir="$2"
  run_logged "${LOG_DIR}/${name}_analyze_distribution.log" \
    python scripts/analyze_sample_outputs.py \
      --input-jsonl "${sample_dir}/raw_generations.jsonl" \
      --failure-jsonl "${sample_dir}/failure_cases.jsonl" \
      --output-json "${NOTES_DIR}/${name}_distribution.json" \
      --output-md "${NOTES_DIR}/${name}_distribution.md"
  run_logged "${LOG_DIR}/${name}_analyze_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --output-json "${NOTES_DIR}/${name}_composition.json" \
      --output-md "${NOTES_DIR}/${name}_composition.md"
}

SMOKE_DIR="${OUT_DIR}/smoke_sft"
train_sft "smoke32" "${SMOKE_DIR}" --limit-train 32 --limit-val 32 --max-train-steps 8
SMOKE_SAMPLE64="${OUT_DIR}/smoke_sample64"
sample_fixed "smoke64" "${SMOKE_DIR}/final" "${SMOKE_SAMPLE64}" 64
analyze_sample "smoke64" "${SMOKE_SAMPLE64}"

STAGE_DIR="${OUT_DIR}/r5_sft"
train_sft "r5_sft" "${STAGE_DIR}"

SAMPLE256_DIR="${OUT_DIR}/sample256"
sample_fixed "sample256" "${STAGE_DIR}/final" "${SAMPLE256_DIR}" 256
analyze_sample "sample256" "${SAMPLE256_DIR}"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
sample_dir = Path("${SAMPLE256_DIR}")
sample = json.loads((sample_dir / "sample_metrics.json").read_text())
comp = json.loads((notes / "sample256_composition.json").read_text())
dist = json.loads((notes / "sample256_distribution.json").read_text())
raw = comp.get("raw_jsonl", comp)
count = int(raw.get("count") or raw.get("total") or 0)
reasons = raw.get("reason_counts", {})
strict = int(reasons.get("charge_neutral_pauling_valid", 0)) / max(1, count)
single = int(reasons.get("single_element_shortcut", 0)) / max(1, count)
valid_reasons = {"charge_neutral_pauling_valid", "all_metal_shortcut", "single_element_shortcut"}
comp_valid = raw.get("comp_valid_rate")
if comp_valid is None:
    comp_valid = sum(int(reasons.get(reason, 0)) for reason in valid_reasons) / max(1, count)
pbc = raw.get("pbc_equivalent_duplicate_fraction")
if pbc is None:
    pbc_count = int(raw.get("pbc_equivalent_duplicate_count", 0) or 0)
    pbc = pbc_count / max(1, count)
total = int(dist.get("total") or dist.get("raw_record_count") or count)
high_sym = float(dist.get("high_symmetry_coord_fraction_mean") or 1.0)
all_lengths_equal = int(dist.get("records_all_lengths_equal") or 0) / max(1, total)
gate = {
  "sample_metrics": {
    "parse_rate": sample.get("parse_rate"),
    "graph_acceptance": sample.get("graph_acceptance_rate"),
    "valid_array_count": sample.get("valid_array_count"),
  },
  "composition": {
    "comp_valid": comp_valid,
    "strict_valid": strict,
    "single_element": single,
    "pbc_duplicate": pbc,
  },
  "geometry": {
    "high_sym_coord_mean": high_sym,
    "a_eq_b_eq_c": all_lengths_equal,
  },
  "baseline": {
    "comp_valid": float("${BASELINE_COMP_VALID_256}"),
    "strict_valid": float("${BASELINE_STRICT_VALID_256}"),
    "high_sym_coord_mean": float("${BASELINE_HIGH_SYM_256}"),
  },
  "thresholds": {
    "parse_rate": 0.98,
    "graph_acceptance": 0.95,
    "comp_valid_min": float("${BASELINE_COMP_VALID_256}") - 0.01,
    "strict_valid_min": float("${BASELINE_STRICT_VALID_256}") - 0.02,
    "single_element_max": 0.05,
    "pbc_duplicate_max": 0.0,
    "high_sym_less_than_baseline": float("${BASELINE_HIGH_SYM_256}"),
  },
}
failures = []
if float(sample.get("parse_rate") or 0) < 0.98: failures.append("parse<0.98")
if float(sample.get("graph_acceptance_rate") or 0) < 0.95: failures.append("graph<0.95")
if comp_valid < float("${BASELINE_COMP_VALID_256}") - 0.01: failures.append("comp_valid<R2-1pt")
if strict < float("${BASELINE_STRICT_VALID_256}") - 0.02: failures.append("strict_valid<R2-2pt")
if single > 0.05: failures.append("single_element>5%")
if pbc > 0.0: failures.append("PBC_duplicate>0")
if high_sym >= float("${BASELINE_HIGH_SYM_256}"): failures.append("high_sym_not_below_R2")
gate["failures"] = failures
gate["passed"] = not failures
(notes / "sample256_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\\n")
print(json.dumps(gate, indent=2, sort_keys=True))
PY

if [ "${FORCE_1000}" != "1" ]; then
  set +e
  python - <<PY
import json, sys
from pathlib import Path
gate = json.loads(Path("${NOTES_DIR}/sample256_gate.json").read_text())
sys.exit(0 if gate.get("passed") else 7)
PY
  status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "R5 failed 256 gate; stopping before 1000."
    exit 0
  fi
fi

if [ "${RUN_1000_IF_PASS}" != "1" ]; then
  echo "RUN_1000_IF_PASS!=1; stopping after 256 gate."
  exit 0
fi

SAMPLE1000_DIR="${OUT_DIR}/sample1000"
sample_fixed "sample1000" "${STAGE_DIR}/final" "${SAMPLE1000_DIR}" "${MAX_ATTEMPTS}" \
  --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
  --max-attempts "${MAX_ATTEMPTS}"
analyze_sample "sample1000" "${SAMPLE1000_DIR}"

REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/sun1000"
mkdir -p "${REFINED1000_DIR}" "${SUN1000_DIR}"
next_port
run_logged "${LOG_DIR}/refine1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${SAMPLE1000_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED1000_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps "${DIFF_STEPS}" \
    --max-proposals "${TARGET_GRAPH_SUCCESS}"

REFINED_PT="${REFINED1000_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt"
run_logged "${LOG_DIR}/crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${REFINED1000_DIR}" \
    --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"
run_logged "${LOG_DIR}/composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
    --refined-pt "${REFINED_PT}" \
    --refined-world-size 2 \
    --output-json "${NOTES_DIR}/composition1000.json" \
    --output-md "${NOTES_DIR}/composition1000.md"

run_logged "${LOG_DIR}/convert_refined1000_extxyz.log" \
  python scripts/convert_crysllmgen_pt_to_extxyz.py \
    --input-pt "${REFINED_PT}" \
    --output-extxyz "${SUN1000_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN1000_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/mattergen_sun1000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json"
  --structures-output-path "${SUN1000_DIR}/relaxed.extxyz"
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json"
  --relax-failures-json "${NOTES_DIR}/mattergen_sun1000_relax_failures.json"
  --unsupported-failures-json "${NOTES_DIR}/mattergen_sun1000_unsupported_failures.json"
  --metric-errors-json "${NOTES_DIR}/mattergen_sun1000_metric_errors.json"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi
run_logged "${LOG_DIR}/mattergen_sun1000.log" \
  "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

run_logged "${LOG_DIR}/mattergen_threshold_analysis.log" \
  python scripts/analyze_mattergen_sun_detailed.py \
    --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
    --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
    --label "${RUN_ID}" \
    --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
    --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"

run_logged "${LOG_DIR}/final_conditional_diagnostics.log" \
  python scripts/analyze_r5_conditional_buckets.py \
    --run "R2=runs/20260527_semalign_selfimprove_r2" \
    --run "R5=${RUN_DIR}" \
    --output-json "${NOTES_DIR}/final_r5_conditional_diagnostics.json" \
    --output-md "${NOTES_DIR}/final_r5_conditional_diagnostics.md" || true

echo "R5 prompt-side v0 complete: ${RUN_DIR}"

#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260528_r3_e3_physical_header}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
PREV_RUN_ID="${PREV_RUN_ID:-20260527_semalign_selfimprove_r2}"
R2_CHECKPOINT="${R2_CHECKPOINT:-runs/${PREV_RUN_ID}/outputs/stage_b/final}"
E1_RUN_ID="${E1_RUN_ID:-20260528_r3_e1_chemplan_smoke}"
BASE_INPUT_DATA_DIR="${BASE_INPUT_DATA_DIR:-data/dlm_sft/mp_20_sun_self_improve_weighted_${PREV_RUN_ID}}"
FALLBACK_INPUT_DATA_DIR="${FALLBACK_INPUT_DATA_DIR:-data/dlm_sft/mp_20}"
HEADER_DATA_DIR="${HEADER_DATA_DIR:-data/dlm_sft/mp_20_physical_header_v0_${RUN_ID}}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
GPU_COUNT="${GPU_COUNT:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-384}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
STAGE_LR="${STAGE_LR:-2e-6}"
STAGE_WARMUP_STEPS="${STAGE_WARMUP_STEPS:-20}"
PHYSICAL_HEADER_LOSS_WEIGHT="${PHYSICAL_HEADER_LOSS_WEIGHT:-2.5}"
ATOM_COUNT_LOSS_WEIGHT="${ATOM_COUNT_LOSS_WEIGHT:-3.0}"
SLOT_MARKER_LOSS_WEIGHT="${SLOT_MARKER_LOSS_WEIGHT:-0.25}"
EMPTY_SLOT_LOSS_WEIGHT="${EMPTY_SLOT_LOSS_WEIGHT:-0.20}"
NONEMPTY_SLOT_LOSS_WEIGHT="${NONEMPTY_SLOT_LOSS_WEIGHT:-2.5}"
LATE_NONEMPTY_SLOT_LOSS_WEIGHT="${LATE_NONEMPTY_SLOT_LOSS_WEIGHT:-4.0}"
COORDINATE_LOSS_WEIGHT="${COORDINATE_LOSS_WEIGHT:-1.1}"
PAD_COORDINATE_LOSS_WEIGHT="${PAD_COORDINATE_LOSS_WEIGHT:-0.10}"
MIN_COMP_VALID_256="${MIN_COMP_VALID_256:-0.88}"
MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256:-0.55}"
MAX_SINGLE_ELEMENT_256="${MAX_SINGLE_ELEMENT_256:-0.10}"
MAX_PBC_DUPLICATE_256="${MAX_PBC_DUPLICATE_256:-0.0}"
MAX_HIGH_SYM_COORD_MEAN_256="${MAX_HIGH_SYM_COORD_MEAN_256:-0.64}"
MAX_A_EQ_B_EQ_C_256="${MAX_A_EQ_B_EQ_C_256:-0.80}"
RUN_E0="${RUN_E0:-1}"
RUN_E1="${RUN_E1:-1}"
RUN_E3="${RUN_E3:-1}"
RUN_1000_IF_PASS="${RUN_1000_IF_PASS:-1}"
FORCE_1000="${FORCE_1000:-0}"
MASTER_PORT_BASE="${MASTER_PORT:-$((27000 + (${SLURM_JOB_ID:-0} % 20000)))}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [ ! -d "${R2_CHECKPOINT}" ]; then
  echo "R2_CHECKPOINT does not exist: ${R2_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -f "${R2_CHECKPOINT}/adapter_config.json" ] && [ ! -f "${R2_CHECKPOINT}/config.json" ]; then
  echo "R2_CHECKPOINT is not a recognizable checkpoint directory: ${R2_CHECKPOINT}" >&2
  exit 2
fi
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}"

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
  "route": "R3 HPG-DLM physical-header-first",
  "r2_checkpoint": "${R2_CHECKPOINT}",
  "e1_run_id": "${E1_RUN_ID}",
  "model_path": "${MODEL_PATH}",
  "base_input_data_dir": "${BASE_INPUT_DATA_DIR}",
  "header_data_dir": "${HEADER_DATA_DIR}",
  "temperature": float("${TEMPERATURE}"),
  "nproc_per_node": int("${NPROC_PER_NODE}"),
  "run_e0": bool(int("${RUN_E0}")),
  "run_e1": bool(int("${RUN_E1}")),
  "run_e3": bool(int("${RUN_E3}")),
  "no_10000": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [ "${RUN_E0}" = "1" ]; then
  run_logged "${LOG_DIR}/e0_overlap_diagnostics.log" \
    python scripts/compare_sun_overlap_diagnostics.py \
      --run "R2=runs/20260527_semalign_selfimprove_r2" \
      --run "ABL1=runs/20260528_stcompress_abl1_refined1000" \
      --run "lowlr5=runs/20260524_mattergen_sun_eval/notes/lowrl5_summary.json" \
      --run "sftbest=runs/20260524_mattergen_sun_eval/notes/sftbest_summary.json" \
      --output-json "${NOTES_DIR}/e0_overlap_diagnostics.json" \
      --output-md "${NOTES_DIR}/e0_overlap_diagnostics.md"
fi

if [ "${RUN_E1}" = "1" ]; then
  RUN_ID="${E1_RUN_ID}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  PREV_BEST_CHECKPOINT="${R2_CHECKPOINT}" \
  PLAN_ROW_FRACTION=0.20 \
  RUN_STAGE_B=0 \
  FORCE_1000=0 \
  STOP_AFTER_256=1 \
  GPU_COUNT="${GPU_COUNT}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MIN_STRICT_VALID_256="${MIN_STRICT_VALID_256}" \
  SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE}" \
  bash scripts/a800/run_mp20_chemplan_bridge_r3.sh
fi

if [ "${RUN_E3}" != "1" ]; then
  echo "RUN_E3!=1; stopping after diagnostics."
  exit 0
fi

if [ ! -f "${BASE_INPUT_DATA_DIR}/train.jsonl" ]; then
  echo "BASE_INPUT_DATA_DIR=${BASE_INPUT_DATA_DIR} missing; falling back to ${FALLBACK_INPUT_DATA_DIR}."
  BASE_INPUT_DATA_DIR="${FALLBACK_INPUT_DATA_DIR}"
fi
if [ ! -f "${BASE_INPUT_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_INPUT_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

run_logged "${LOG_DIR}/unit_tests.log" \
  python -m unittest tests.test_fixed_slot_physical_header tests.test_llada_generation_masks

if [ "${REBUILD_HEADER_DATA:-0}" = "1" ]; then
  rm -rf "${HEADER_DATA_DIR}"
fi
if [ ! -f "${HEADER_DATA_DIR}/train.jsonl" ]; then
  run_logged "${LOG_DIR}/build_header_data.log" \
    python scripts/build_fixed_slot_physical_header_sft_data.py \
      --input-dir "${BASE_INPUT_DATA_DIR}" \
      --output-dir "${HEADER_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi
cp "${HEADER_DATA_DIR}/stats.json" "${NOTES_DIR}/input_header_stats.json" || true
cp "${HEADER_DATA_DIR}/result.md" "${NOTES_DIR}/input_header_data_result.md" || true

train_header() {
  local name="$1"
  local output_dir="$2"
  local checkpoint="$3"
  local extra_limits=()
  if [ "$#" -gt 3 ]; then
    extra_limits=("${@:4}")
  fi
  next_port
  run_logged "${LOG_DIR}/${name}_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --representation fixed_slot_physical_header \
      --data-dir "${HEADER_DATA_DIR}" \
      --output-dir "${output_dir}" \
      --max-length "${SFT_MAX_LENGTH}" \
      --epochs 1 \
      --batch-size "${SFT_BATCH_SIZE}" \
      --grad-accum "${SFT_GRAD_ACCUM}" \
      --lr "${STAGE_LR}" \
      --lr-scheduler cosine \
      --warmup-steps "${STAGE_WARMUP_STEPS}" \
      --min-lr-ratio 0.2 \
      --physical-header-loss-weight "${PHYSICAL_HEADER_LOSS_WEIGHT}" \
      --atom-count-loss-weight "${ATOM_COUNT_LOSS_WEIGHT}" \
      --slot-marker-loss-weight "${SLOT_MARKER_LOSS_WEIGHT}" \
      --empty-slot-loss-weight "${EMPTY_SLOT_LOSS_WEIGHT}" \
      --nonempty-slot-loss-weight "${NONEMPTY_SLOT_LOSS_WEIGHT}" \
      --late-nonempty-slot-loss-weight "${LATE_NONEMPTY_SLOT_LOSS_WEIGHT}" \
      --coordinate-loss-weight "${COORDINATE_LOSS_WEIGHT}" \
      --pad-coordinate-loss-weight "${PAD_COORDINATE_LOSS_WEIGHT}" \
      --train-prefill-slot-tokens \
      --logging-steps 20 \
      --eval-steps 424 \
      --save-steps 848 \
      --position-diagnostics-steps 424 \
      ${extra_limits[@]+"${extra_limits[@]}"}
}

sample_and_analyze() {
  local name="$1"
  local checkpoint="$2"
  local num_samples="$3"
  local sample_dir="$4"
  local notes_prefix="$5"
  mkdir -p "${sample_dir}" "${notes_prefix}"
  next_port
  run_logged "${LOG_DIR}/${name}_sample.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --representation fixed_slot_physical_header \
      --output-dir "${sample_dir}" \
      --num-samples "${num_samples}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --block-length 1 \
      --temperature "${TEMPERATURE}" \
      --generation-schedule "${GENERATION_SCHEDULE}" \
      --schema-logit-mask \
      --prefill-slot-tokens \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask
  python scripts/analyze_sample_outputs.py \
    --input-jsonl "${sample_dir}/raw_generations.jsonl" \
    --failure-jsonl "${sample_dir}/failure_cases.jsonl" \
    --representation fixed_slot_physical_header \
    --output-json "${notes_prefix}/sample_distribution.json" \
    --output-md "${notes_prefix}/sample_distribution.md"
  python scripts/analyze_composition_validity.py \
    --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_prefix}/sample_composition.json" \
    --output-md "${notes_prefix}/sample_composition.md"
  python scripts/analyze_composition_failure_modes.py \
    --raw-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_prefix}/sample_failure_modes.json" \
    --output-md "${notes_prefix}/sample_failure_modes.md"
}

SMOKE_DIR="${OUT_DIR}/smoke_sft"
train_header "smoke" "${SMOKE_DIR}" "${R2_CHECKPOINT}" --limit-train 32 --limit-val 32
sample_and_analyze "smoke64" "${SMOKE_DIR}/final" 64 "${OUT_DIR}/smoke64" "${NOTES_DIR}/smoke64"

STAGE_DIR="${OUT_DIR}/stage_a"
train_header "stage_a" "${STAGE_DIR}" "${R2_CHECKPOINT}"
sample_and_analyze "stage_a_256" "${STAGE_DIR}/final" 256 "${OUT_DIR}/stage_a_sample256" "${NOTES_DIR}/stage_a"

python scripts/evaluate_mp20_candidate_gate.py \
  --mode smoke256 \
  --sample-metrics "${OUT_DIR}/stage_a_sample256/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/stage_a/sample_composition.json" \
  --composition-key raw_jsonl \
  --min-parse-rate 0.98 \
  --min-graph-acceptance 0.95 \
  --min-comp-valid "${MIN_COMP_VALID_256}" \
  --min-strict-valid "${MIN_STRICT_VALID_256}" \
  --max-single-element "${MAX_SINGLE_ELEMENT_256}" \
  --max-pbc-duplicate "${MAX_PBC_DUPLICATE_256}" \
  --output-json "${NOTES_DIR}/stage_a/sample256_gate.json" || true

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}") / "stage_a"
dist = json.loads((notes / "sample_distribution.json").read_text())
gate = json.loads((notes / "sample256_gate.json").read_text())
total = max(1, int(dist.get("total") or 0))
high = float(dist.get("high_symmetry_coord_fraction_mean") or 0.0)
abc = float(dist.get("records_all_lengths_equal") or 0) / total
all90 = float(dist.get("records_all_angles_90") or 0) / total
passed = bool(gate.get("passed")) and high <= float("${MAX_HIGH_SYM_COORD_MEAN_256}") and abc <= float("${MAX_A_EQ_B_EQ_C_256}")
payload = {
  "passed": passed,
  "base_gate_passed": bool(gate.get("passed")),
  "high_symmetry_coord_fraction_mean": high,
  "a_eq_b_eq_c": abc,
  "all_90": all90,
  "max_high_symmetry_coord_fraction_mean": float("${MAX_HIGH_SYM_COORD_MEAN_256}"),
  "max_a_eq_b_eq_c": float("${MAX_A_EQ_B_EQ_C_256}"),
  "base_gate": gate,
}
(notes / "sample256_physical_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

PASSED_256="$(python - <<PY
import json
from pathlib import Path
payload=json.loads((Path("${NOTES_DIR}")/"stage_a"/"sample256_physical_gate.json").read_text())
print("1" if payload.get("passed") else "0")
PY
)"
if [ "${PASSED_256}" != "1" ] && [ "${FORCE_1000}" != "1" ]; then
  echo "Physical-header stage_a failed 256 gate; stopping before 1000."
  exit 0
fi
if [ "${RUN_1000_IF_PASS}" != "1" ]; then
  echo "RUN_1000_IF_PASS!=1; stopping after 256."
  exit 0
fi

SAMPLE1000_DIR="${OUT_DIR}/sample1000"
REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/mattergen_sun1000"
mkdir -p "${SAMPLE1000_DIR}" "${REFINED1000_DIR}" "${SUN1000_DIR}"

next_port
run_logged "${LOG_DIR}/sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${STAGE_DIR}/final" \
    --representation fixed_slot_physical_header \
    --output-dir "${SAMPLE1000_DIR}" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --num-samples "${MAX_ATTEMPTS}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --block-length 1 \
    --temperature "${TEMPERATURE}" \
    --generation-schedule "${GENERATION_SCHEDULE}" \
    --schema-logit-mask \
    --prefill-slot-tokens \
    --atom-count-grammar-mask \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE1000_DIR}/failure_cases.jsonl" \
  --representation fixed_slot_physical_header \
  --output-json "${NOTES_DIR}/sample1000_distribution.json" \
  --output-md "${NOTES_DIR}/sample1000_distribution.md"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample1000_composition_raw.json" \
  --output-md "${NOTES_DIR}/sample1000_composition_raw.md"

TARGET_REACHED="$(python - <<PY
import json
from pathlib import Path
print("1" if json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()).get("target_reached") else "0")
PY
)"
if [ "${TARGET_REACHED}" != "1" ]; then
  echo "1000 graph-valid target not reached; stopping before refinement/SUN."
  exit 0
fi

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
python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED1000_DIR}" \
  --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --refined-pt "${REFINED_PT}" \
  --refined-world-size 2 \
  --output-json "${NOTES_DIR}/composition1000.json" \
  --output-md "${NOTES_DIR}/composition1000.md"
python scripts/evaluate_mp20_candidate_gate.py \
  --mode refined1000 \
  --sample-metrics "${SAMPLE1000_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/composition1000.json" \
  --composition-key refined_pt \
  --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
  --max-single-element 0.10 \
  --max-pbc-duplicate 0.0 \
  --output-json "${NOTES_DIR}/refined1000_gate.json" || true

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
  --relax-max-steps "${RELAX_MAX_STEPS:-500}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi
run_logged "${LOG_DIR}/mattergen_sun1000.log" \
  "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

python scripts/analyze_mattergen_sun_detailed.py \
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --label "${RUN_ID}" \
  --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
  --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"

python scripts/compare_sun_overlap_diagnostics.py \
  --run "R2=runs/20260527_semalign_selfimprove_r2" \
  --run "physical_header=${RUN_DIR}" \
  --output-json "${NOTES_DIR}/final_overlap_diagnostics.json" \
  --output-md "${NOTES_DIR}/final_overlap_diagnostics.md"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
def read(name):
    path = notes / name
    return json.loads(path.read_text()) if path.exists() else {}
payload = {
  "run_id": "${RUN_ID}",
  "checkpoint": "${STAGE_DIR}/final",
  "sample256_gate": read("stage_a/sample256_physical_gate.json"),
  "sample1000": json.loads(Path("${SAMPLE1000_DIR}/sample_metrics.json").read_text()),
  "crysllmgen": read("crysllmgen_metrics1000.json"),
  "composition1000": read("composition1000.json"),
  "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
  "no_10000": True,
}
(notes / "result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "R3 physical-header run complete: ${RUN_DIR}"

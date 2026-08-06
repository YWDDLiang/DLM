#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260526_stalign_restart_r0}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
WEIGHTED_DATA_DIR="${WEIGHTED_DATA_DIR:-data/dlm_sft/mp_20_ehull_weighted_r0}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}"

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
  "model_path": "${MODEL_PATH}",
  "base_data_dir": "${BASE_DATA_DIR}",
  "weighted_data_dir": "${WEIGHTED_DATA_DIR}",
  "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
  "mattergen_python": "${MATTERGEN_PYTHON}",
  "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
  "temperature": float("${TEMPERATURE}"),
  "generation_schedule": "${GENERATION_SCHEDULE}",
  "stage": "R0 preflight/smoke",
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/preflight_compile_tests.log" \
  python -m unittest \
    tests.test_mp20_ehull_weighted_sft_data \
    tests.test_strict_sun_self_improving_buffer \
    tests.test_llada_sft_weights \
    tests.test_llada_generation_masks \
    tests.test_composition_validity

python -m py_compile \
  scripts/build_mp20_ehull_weighted_sft_data.py \
  scripts/build_strict_sun_self_improving_buffer.py \
  scripts/llada_sft.py \
  scripts/sample_llada_crystals.py \
  scripts/run_mattergen_sun_eval.py

python - <<PY
import json
from pathlib import Path
status = {}
for key, path in {
  "model_path": "${MODEL_PATH}",
  "base_data_dir": "${BASE_DATA_DIR}",
  "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
  "mattergen_python": "${MATTERGEN_PYTHON}",
  "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
}.items():
  p = Path(path)
  status[key] = {"path": path, "exists": p.exists(), "is_dir": p.is_dir()}
missing = [key for key, value in status.items() if not value["exists"]]
Path("${NOTES_DIR}/preflight_paths.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
if missing:
  raise SystemExit("missing_preflight_paths:" + ",".join(missing))
PY

if [ ! -f "${BASE_DATA_DIR}/train.jsonl" ] || [ ! -f "${BASE_DATA_DIR}/vocab_tokens.txt" ]; then
  run_logged "${LOG_DIR}/build_fixed_slot_data.log" \
    python scripts/build_crystal_sft_data.py \
      --input-dir reference/crysllmgen/data/mp_20 \
      --output-dir "${BASE_DATA_DIR}" \
      --tokenizer-path "${MODEL_PATH}" \
      --answer-separator ""
fi

run_logged "${LOG_DIR}/build_ehull_weighted_data.log" \
  python scripts/build_mp20_ehull_weighted_sft_data.py \
    --base-dir "${BASE_DATA_DIR}" \
    --csv-dir reference/crysllmgen/data/mp_20 \
    --output-dir "${WEIGHTED_DATA_DIR}" \
    --extra-fraction 0.15 \
    --max-formula-repeats 8 \
    --max-chemsys-repeats 64

SMOKE_DIR="${OUT_DIR}/sft_smoke32"
run_logged "${LOG_DIR}/sft_smoke32.log" \
  torchrun --nproc_per_node=2 --master_port "${MASTER_PORT}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${WEIGHTED_DATA_DIR}" \
    --output-dir "${SMOKE_DIR}" \
    --limit-train 32 \
    --limit-val 32 \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --lr 5e-5 \
    --lr-scheduler cosine \
    --warmup-steps 2 \
    --min-lr-ratio 0.2 \
    --atom-count-loss-weight 3.0 \
    --slot-marker-loss-weight 0.25 \
    --empty-slot-loss-weight 0.5 \
    --nonempty-slot-loss-weight 2.0 \
    --late-nonempty-slot-loss-weight 4.0 \
    --coordinate-loss-weight 1.0 \
    --pad-coordinate-loss-weight 0.2 \
    --semantic-init-element-tokens \
    --train-prefill-slot-tokens \
    --logging-steps 1 \
    --eval-steps 8 \
    --save-steps 1000 \
    --position-diagnostics-steps 8

SAMPLE_DIR="${OUT_DIR}/sample64"
run_logged "${LOG_DIR}/sample64.log" \
  torchrun --nproc_per_node=2 --master_port "$((MASTER_PORT + 1))" scripts/sample_llada_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${SMOKE_DIR}/final" \
    --output-dir "${SAMPLE_DIR}" \
    --num-samples 64 \
    --batch-size 8 \
    --block-length 1 \
    --temperature "${TEMPERATURE}" \
    --generation-schedule "${GENERATION_SCHEDULE}" \
    --schema-logit-mask \
    --prefill-slot-tokens \
    --atom-count-grammar-mask \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/sample64_distribution.json" \
  --output-md "${NOTES_DIR}/sample64_distribution.md"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample64_composition.json" \
  --output-md "${NOTES_DIR}/sample64_composition.md"
python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/sample64_failure_modes.json" \
  --output-md "${NOTES_DIR}/sample64_failure_modes.md"

python - <<PY
import json
from pathlib import Path
sample = json.loads(Path("${SAMPLE_DIR}/sample_metrics.json").read_text())
comp = json.loads(Path("${NOTES_DIR}/sample64_composition.json").read_text())
payload = {
  "passed": sample.get("parse_rate", 0) >= 0.98 and sample.get("graph_acceptance_rate", 0) >= 0.95,
  "sample_metrics": sample,
  "composition": comp.get("raw_jsonl", comp),
  "weighted_data_summary": "${WEIGHTED_DATA_DIR}/ehull_weight_summary.json",
  "semantic_report": "${SMOKE_DIR}/element_special_token_alignment_report.json",
}
Path("${NOTES_DIR}/r0_smoke_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

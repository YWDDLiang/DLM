#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_crysllmgen_text}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GEN_LENGTH="${GEN_LENGTH:-320}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_LENGTH="${MAX_LENGTH:-768}"
CRYSLLMGEN_TEXT_PROMPT_VERSION="${CRYSLLMGEN_TEXT_PROMPT_VERSION:-crysllmgen_text_v1_single_pass}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
BRANCH_DIR="${RUN_DIR}/outputs/crysllmgen_text"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${BRANCH_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((BASE_MASTER_PORT + PORT_OFFSET))
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

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    crystal_dlm/crysllmgen_text.py \
    scripts/build_crysllmgen_text_sft_data.py \
    scripts/sample_llada_crysllmgen_text.py \
    scripts/llada_sft.py \
    scripts/analyze_composition_validity.py

DATA_READY=$(python - <<PY
import json
from pathlib import Path
data_dir = Path("${DATA_DIR}")
required = [data_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
required += [data_dir / "stats.json", data_dir / "_SUCCESS"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    print(0)
    raise SystemExit
try:
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
ok = stats.get("representation") == "crysllmgen_text"
ok = ok and stats.get("prompt_version") == "${CRYSLLMGEN_TEXT_PROMPT_VERSION}"
ok = ok and all(int(stats.get("splits", {}).get(split, {}).get("rows_written", 0)) > 0 for split in ("train", "val", "test"))
print(1 if ok else 0)
PY
)

if [[ "${DATA_READY}" != "1" ]]; then
  run_logged "${LOG_DIR}/build_crysllmgen_text_data.log" \
    python scripts/build_crysllmgen_text_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --skip-graph-validation
else
  echo "CrysLLMGen text data is already complete at ${DATA_DIR}; reusing it." | tee -a "${LOG_DIR}/build_crysllmgen_text_data.log"
fi

python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${DATA_DIR}/stats.json").read_text(encoding="utf-8"))
payload = {
    "run_id": "${RUN_ID}",
    "representation": "crysllmgen_text",
    "model_path": "${MODEL_PATH}",
    "data_dir": "${DATA_DIR}",
    "temperature": float("${TEMPERATURE}"),
    "gen_length": int("${GEN_LENGTH}"),
    "max_length": int("${MAX_LENGTH}"),
    "sampling_ablation": ["block_length=1", "block_length=4"],
    "data_stats": {
        "max_answer_model_length": stats.get("max_answer_model_length"),
        "max_prompt_model_length": stats.get("max_prompt_model_length"),
        "max_length_recommended": stats.get("max_length_recommended"),
        "train_rows": stats.get("splits", {}).get("train", {}).get("rows_written"),
        "prompt_version": stats.get("prompt_version"),
    },
}
Path("${NOTES_DIR}/crysllmgen_text_smoke_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

next_port
run_logged "${LOG_DIR}/stage0_smoke_train.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --representation crysllmgen_text \
    --skip-data-vocab-resize \
    --output-dir "${BRANCH_DIR}/stage0_smoke" \
    --max-length "${MAX_LENGTH}" \
    --limit-train 32 \
    --limit-val 32 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --lr 1e-4 \
    --lr-scheduler cosine \
    --warmup-steps 2 \
    --min-lr-ratio 0.2 \
    --save-steps 32 \
    --eval-steps 16 \
    --modules-to-save "" \
    --save-embedding-layers false

sample_block() {
  local block_length="$1"
  local name="stage0_sample64_b${block_length}"
  next_port
  run_logged "${LOG_DIR}/${name}.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_crysllmgen_text.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${BRANCH_DIR}/stage0_smoke/final" \
      --output-dir "${BRANCH_DIR}/${name}" \
      --num-samples 64 \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --gen-length "${GEN_LENGTH}" \
      --block-length "${block_length}" \
      --temperature "${TEMPERATURE}"
  local raw_pt_args=()
  local composition_key="raw_jsonl"
  if [[ -s "${BRANCH_DIR}/${name}/raw_dlm_samples.pt" ]]; then
    raw_pt_args=(--raw-pt "${BRANCH_DIR}/${name}/raw_dlm_samples.pt")
    composition_key="raw_pt"
  fi
  run_logged "${LOG_DIR}/${name}_composition.log" \
    python scripts/analyze_composition_validity.py \
      "${raw_pt_args[@]}" \
      --raw-generations-jsonl "${BRANCH_DIR}/${name}/raw_generations.jsonl" \
      --representation crysllmgen_text \
      --output-json "${NOTES_DIR}/${name}_composition.json" \
      --output-md "${NOTES_DIR}/${name}_composition.md"
  python - <<PY
import json
from pathlib import Path
sample = json.loads(Path("${BRANCH_DIR}/${name}/sample_metrics.json").read_text(encoding="utf-8"))
comp_path = Path("${NOTES_DIR}/${name}_composition.json")
comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}
summary = {
    "name": "${name}",
    "block_length": int("${block_length}"),
    "sample_metrics": sample,
    "composition_key": "${composition_key}",
    "composition_summary_path": str(comp_path),
    "raw_pt_comp_valid": comp.get("raw_pt", {}).get("comp_valid_rate"),
    "raw_jsonl_comp_valid": comp.get("raw_jsonl", {}).get("comp_valid_rate"),
}
Path("${NOTES_DIR}/${name}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

sample_block 1
sample_block 4

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
items = []
for path in sorted(notes.glob("stage0_sample64_b*_summary.json")):
    items.append(json.loads(path.read_text(encoding="utf-8")))
lines = ["# CrysLLMGen Text Smoke Sampling Ablation", ""]
for item in items:
    m = item["sample_metrics"]
    lines.append(
        f"- {item['name']}: block={item['block_length']}, "
        f"parse={m.get('parse_rate', 0):.4f}, graph={m.get('graph_acceptance_rate', 0):.4f}, "
        f"time_sec={m.get('time_sec', 0):.1f}, raw_pt_comp={item.get('raw_pt_comp_valid')}"
    )
(notes / "crysllmgen_text_smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(notes / "crysllmgen_text_smoke_summary.json").write_text(json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

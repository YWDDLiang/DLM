#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_fixed_plain}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-2e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-256}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS_1000="${MAX_ATTEMPTS_1000:-1800}"
TEMPERATURE="${TEMPERATURE:-0.7}"
COUNT_GEN_LENGTH="${COUNT_GEN_LENGTH:-8}"
COUNT_BLOCK_LENGTH="${COUNT_BLOCK_LENGTH:-4}"
LATTICE_GEN_LENGTH="${LATTICE_GEN_LENGTH:-48}"
LATTICE_BLOCK_LENGTH="${LATTICE_BLOCK_LENGTH:-8}"
ELEMENTS_TOKENS_PER_SITE="${ELEMENTS_TOKENS_PER_SITE:-6}"
ELEMENTS_EXTRA_TOKENS="${ELEMENTS_EXTRA_TOKENS:-8}"
ELEMENTS_BLOCK_LENGTH="${ELEMENTS_BLOCK_LENGTH:-4}"
COORDS_TOKENS_PER_SITE="${COORDS_TOKENS_PER_SITE:-12}"
COORDS_EXTRA_TOKENS="${COORDS_EXTRA_TOKENS:-8}"
COORDS_BLOCK_LENGTH="${COORDS_BLOCK_LENGTH:-8}"
COORDS_MAX_GEN_LENGTH="${COORDS_MAX_GEN_LENGTH:-288}"
MIN_PARSE_RATE="${MIN_PARSE_RATE:-0.98}"
MIN_GRAPH_ACCEPTANCE="${MIN_GRAPH_ACCEPTANCE:-0.95}"
MIN_COMP_VALID="${MIN_COMP_VALID:-0.88}"
MIN_STRICT_VALID="${MIN_STRICT_VALID:-0.30}"
MAX_SINGLE_ELEMENT="${MAX_SINGLE_ELEMENT:-0.10}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
BRANCH_DIR="${RUN_DIR}/outputs/fixed_plain"
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
    crystal_dlm/fixed_plain.py \
    scripts/build_fixed_plain_sft_data.py \
    scripts/sample_llada_fixed_plain.py \
    scripts/llada_sft.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_mp20_candidate_gate.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py

run_logged "${LOG_DIR}/preflight_unittest.log" \
  python -m unittest tests.test_fixed_plain

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
ok = stats.get("representation") == "fixed_plain"
ok = ok and stats.get("prompt_version") == "fixed_plain_v1_staged"
ok = ok and all(int(stats.get("splits", {}).get(split, {}).get("rows_written", 0)) > 0 for split in ("train", "val", "test"))
print(1 if ok else 0)
PY
)

if [[ "${DATA_READY}" != "1" ]]; then
  run_logged "${LOG_DIR}/build_fixed_plain_data.log" \
    python scripts/build_fixed_plain_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --skip-graph-validation
else
  echo "fixed-plain data is already complete at ${DATA_DIR}; reusing it." | tee -a "${LOG_DIR}/build_fixed_plain_data.log"
fi

MAX_LENGTH=$(python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${DATA_DIR}/stats.json").read_text(encoding="utf-8"))
recommended = int(stats.get("max_length_recommended") or 512)
print(min(768, max(256, recommended + 32)))
PY
)

python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${DATA_DIR}/stats.json").read_text(encoding="utf-8"))
payload = {
    "run_id": "${RUN_ID}",
    "model_path": "${MODEL_PATH}",
    "data_dir": "${DATA_DIR}",
    "representation": "fixed_plain",
    "train": {
        "epochs": 1,
        "lr": float("${TRAIN_LR}"),
        "lr_scheduler": "cosine",
        "warmup_steps": int("${TRAIN_WARMUP_STEPS}"),
        "batch_size": int("${TRAIN_BATCH_SIZE}"),
        "grad_accum": int("${TRAIN_GRAD_ACCUM}"),
        "max_length": int("${MAX_LENGTH}"),
    },
    "sampling": {
        "temperature": float("${TEMPERATURE}"),
        "count": [int("${COUNT_GEN_LENGTH}"), int("${COUNT_BLOCK_LENGTH}")],
        "lattice": [int("${LATTICE_GEN_LENGTH}"), int("${LATTICE_BLOCK_LENGTH}")],
        "elements_tokens_per_site": int("${ELEMENTS_TOKENS_PER_SITE}"),
        "elements_block_length": int("${ELEMENTS_BLOCK_LENGTH}"),
        "coords_tokens_per_site": int("${COORDS_TOKENS_PER_SITE}"),
        "coords_block_length": int("${COORDS_BLOCK_LENGTH}"),
    },
    "gate": {
        "min_parse_rate": float("${MIN_PARSE_RATE}"),
        "min_graph_acceptance": float("${MIN_GRAPH_ACCEPTANCE}"),
        "min_comp_valid": float("${MIN_COMP_VALID}"),
        "min_strict_valid": float("${MIN_STRICT_VALID}"),
        "max_single_element": float("${MAX_SINGLE_ELEMENT}"),
    },
    "data_stats": {
        "max_answer_model_length": stats.get("max_answer_model_length"),
        "max_prompt_model_length": stats.get("max_prompt_model_length"),
        "max_length_recommended": stats.get("max_length_recommended"),
        "train_rows": stats.get("splits", {}).get("train", {}).get("rows_written"),
        "module_counts": stats.get("splits", {}).get("train", {}).get("module_counts"),
    },
}
Path("${NOTES_DIR}/fixed_plain_one_epoch_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

next_port
run_logged "${LOG_DIR}/sft_epoch1.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --representation fixed_plain \
    --skip-data-vocab-resize \
    --output-dir "${BRANCH_DIR}/sft_epoch1" \
    --max-length "${MAX_LENGTH}" \
    --epochs 1 \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --grad-accum "${TRAIN_GRAD_ACCUM}" \
    --lr "${TRAIN_LR}" \
    --lr-scheduler cosine \
    --warmup-steps "${TRAIN_WARMUP_STEPS}" \
    --min-lr-ratio 0.2 \
    --logging-steps 20 \
    --eval-steps 500 \
    --eval-max-batches 50 \
    --save-steps 999999 \
    --position-diagnostics-steps 1000 \
    --dataloader-num-workers 2 \
    --modules-to-save "" \
    --save-embedding-layers false \
    --fixed-plain-count-loss-weight 3.0 \
    --fixed-plain-lattice-loss-weight 1.0 \
    --fixed-plain-elements-loss-weight 2.0 \
    --fixed-plain-coords-loss-weight 1.1

run_logged "${LOG_DIR}/sft_epoch1_loss_check.log" \
  python - <<PY
import json, math
from pathlib import Path
log_path = Path("${BRANCH_DIR}/sft_epoch1/training_log.jsonl")
train, evals, diagnostics = [], [], []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("event") == "train":
        train.append(row)
    elif row.get("event") == "eval":
        evals.append(row)
    elif row.get("event") == "position_diagnostics":
        diagnostics.append(row)
bad = [row for row in train if not math.isfinite(float(row.get("loss", float("nan"))))]
losses = [float(row["loss"]) for row in train if math.isfinite(float(row.get("loss", float("nan"))))]
summary = {
    "train_events": len(train),
    "eval_events": len(evals),
    "diagnostic_events": len(diagnostics),
    "first_train": train[0] if train else None,
    "last_train": train[-1] if train else None,
    "last_eval": evals[-1] if evals else None,
    "last_position_diagnostics": diagnostics[-1] if diagnostics else None,
    "nonfinite_loss_count": len(bad),
    "max_train_loss": max(losses) if losses else None,
    "min_train_loss": min(losses) if losses else None,
}
Path("${NOTES_DIR}/sft_epoch1_loss_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if bad:
    raise SystemExit("non-finite training loss detected")
PY

next_port
run_logged "${LOG_DIR}/sample256.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_fixed_plain.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BRANCH_DIR}/sft_epoch1/final" \
    --output-dir "${BRANCH_DIR}/sample256" \
    --num-samples "${SMOKE_SAMPLES}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --count-gen-length "${COUNT_GEN_LENGTH}" \
    --count-block-length "${COUNT_BLOCK_LENGTH}" \
    --lattice-gen-length "${LATTICE_GEN_LENGTH}" \
    --lattice-block-length "${LATTICE_BLOCK_LENGTH}" \
    --elements-tokens-per-site "${ELEMENTS_TOKENS_PER_SITE}" \
    --elements-extra-tokens "${ELEMENTS_EXTRA_TOKENS}" \
    --elements-block-length "${ELEMENTS_BLOCK_LENGTH}" \
    --coords-tokens-per-site "${COORDS_TOKENS_PER_SITE}" \
    --coords-extra-tokens "${COORDS_EXTRA_TOKENS}" \
    --coords-block-length "${COORDS_BLOCK_LENGTH}" \
    --coords-max-gen-length "${COORDS_MAX_GEN_LENGTH}"

COMPOSITION_KEY="raw_jsonl"
if [[ -s "${BRANCH_DIR}/sample256/raw_dlm_samples.pt" ]]; then
  COMPOSITION_KEY="raw_pt"
  run_logged "${LOG_DIR}/sample256_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${BRANCH_DIR}/sample256/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${BRANCH_DIR}/sample256/raw_generations.jsonl" \
      --representation fixed_plain \
      --output-json "${NOTES_DIR}/sample256_composition.json" \
      --output-md "${NOTES_DIR}/sample256_composition.md"
else
  run_logged "${LOG_DIR}/sample256_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-generations-jsonl "${BRANCH_DIR}/sample256/raw_generations.jsonl" \
      --representation fixed_plain \
      --output-json "${NOTES_DIR}/sample256_composition.json" \
      --output-md "${NOTES_DIR}/sample256_composition.md"
fi

run_logged "${LOG_DIR}/sample256_gate.log" \
  python scripts/evaluate_mp20_candidate_gate.py \
    --mode smoke256 \
    --sample-metrics "${BRANCH_DIR}/sample256/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/sample256_composition.json" \
    --composition-key "${COMPOSITION_KEY}" \
    --min-parse-rate "${MIN_PARSE_RATE}" \
    --min-graph-acceptance "${MIN_GRAPH_ACCEPTANCE}" \
    --min-comp-valid "${MIN_COMP_VALID}" \
    --min-strict-valid "${MIN_STRICT_VALID}" \
    --max-single-element "${MAX_SINGLE_ELEMENT}" \
    --max-pbc-duplicate 0.0 \
    --output-json "${NOTES_DIR}/sample256_gate.json"

run_logged "${LOG_DIR}/sample256_failure_digest.log" \
  python - <<PY
import json
from collections import Counter
from pathlib import Path
sample_path = Path("${BRANCH_DIR}/sample256/sample_metrics.json")
failure_path = Path("${BRANCH_DIR}/sample256/failure_cases.jsonl")
raw_path = Path("${BRANCH_DIR}/sample256/raw_generations.jsonl")
sample = json.loads(sample_path.read_text(encoding="utf-8"))
failures = []
if failure_path.exists():
    for line in failure_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            failures.append(json.loads(line))
stage_counts = Counter(item.get("stage", "unknown") for item in failures)
reason_counts = Counter(f"{item.get('stage', 'unknown')}:{item.get('reason', 'unknown')}" for item in failures)
raw_examples = []
if raw_path.exists():
    for line in raw_path.read_text(encoding="utf-8").splitlines()[:30]:
        if line.strip():
            row = json.loads(line)
            raw_examples.append({
                "sample_idx": row.get("sample_idx"),
                "parsed": row.get("parsed"),
                "num_atoms": row.get("num_atoms"),
                "reason": row.get("reason"),
                "count_text": row.get("count_text"),
                "lattice_text": row.get("lattice_text"),
                "elements_text": row.get("elements_text"),
                "coords_text_head": (row.get("coords_text") or "")[:300],
            })
payload = {
    "sample_metrics": sample,
    "failure_stage_counts": dict(stage_counts.most_common()),
    "failure_reason_counts": dict(reason_counts.most_common()),
    "failure_examples": failures[:20],
    "raw_examples": raw_examples,
}
Path("${NOTES_DIR}/sample256_failure_digest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
PY

SHOULD_RUN_1000=$(python - <<PY
import json
from pathlib import Path
gate = json.loads(Path("${NOTES_DIR}/sample256_gate.json").read_text(encoding="utf-8"))
metrics = gate.get("metrics", {})
near = (
    metrics.get("parse_rate", 0) >= float("${MIN_PARSE_RATE}")
    and metrics.get("graph_acceptance", 0) >= float("${MIN_GRAPH_ACCEPTANCE}")
    and metrics.get("comp_valid", 0) >= float("${MIN_COMP_VALID}")
    and metrics.get("single_element", 1) <= float("${MAX_SINGLE_ELEMENT}")
)
print(1 if (gate.get("passed") or near) else 0)
PY
)

if [[ "${SHOULD_RUN_1000}" == "1" ]]; then
  next_port
  run_logged "${LOG_DIR}/sample1000.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_fixed_plain.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${BRANCH_DIR}/sft_epoch1/final" \
      --output-dir "${BRANCH_DIR}/sample1000" \
      --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
      --max-attempts "${MAX_ATTEMPTS_1000}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --temperature "${TEMPERATURE}"

  if [[ -s "${BRANCH_DIR}/sample1000/proposal_graphs.pt" ]]; then
    next_port
    run_logged "${LOG_DIR}/refined1000.log" \
      torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${BRANCH_DIR}/sample1000/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${BRANCH_DIR}/refined1000" \
        --diff-steps 800 \
        --max-proposals "${TARGET_GRAPH_SUCCESS}"

    run_logged "${LOG_DIR}/crysllmgen_metrics1000.log" \
      python scripts/run_crysllmgen_metrics.py \
        --root-path "${BRANCH_DIR}/refined1000" \
        --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"
  fi
fi

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
branch = Path("${BRANCH_DIR}")
def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
payload = {
    "config": load(notes / "fixed_plain_one_epoch_config.json"),
    "loss": load(notes / "sft_epoch1_loss_summary.json"),
    "sample256": load(branch / "sample256" / "sample_metrics.json"),
    "composition256": load(notes / "sample256_composition.json"),
    "gate256": load(notes / "sample256_gate.json"),
    "sample1000": load(branch / "sample1000" / "sample_metrics.json"),
    "crysllmgen_metrics1000": load(notes / "crysllmgen_metrics1000.json"),
}
lines = ["# Fixed-Plain Staged One-Epoch Trial", "", "```json", json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "```"]
(notes / "fixed_plain_one_epoch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

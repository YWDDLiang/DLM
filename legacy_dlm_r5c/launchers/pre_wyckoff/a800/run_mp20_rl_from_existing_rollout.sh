#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-${1:-}}"
ROLLOUT_RAW_JSONL="${ROLLOUT_RAW_JSONL:-${2:-}}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:?BEST_CHECKPOINT is required}"
ROLLOUT_RAW_JSONL="${ROLLOUT_RAW_JSONL:?ROLLOUT_RAW_JSONL is required}"

REWARD_MODE="${REWARD_MODE:-comp_valid_priority}"
TEMPERATURE="${TEMPERATURE:-0.7}"
LR="${LR:-1e-7}"
CLIP_EPS="${CLIP_EPS:-0.15}"
BETA="${BETA:-0.03}"
TRACE_SHRINK="${TRACE_SHRINK:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
CEPO_LAMBDA="${CEPO_LAMBDA:-0.15}"
CEPO_CLIP_EPS="${CEPO_CLIP_EPS:-0.15}"
CEPO_ZERO_NON_CREDIT="${CEPO_ZERO_NON_CREDIT:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"

RUN_DIR="runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/outputs/rollout" "${RUN_DIR}/outputs/tracerl" "${RUN_DIR}/outputs/sample256" "${RUN_DIR}/notes"
cp "${ROLLOUT_RAW_JSONL}" "${RUN_DIR}/outputs/rollout/rollout_raw.jsonl"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
(run / "notes" / "run_config.json").write_text(json.dumps({
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "source_rollout_raw_jsonl": "${ROLLOUT_RAW_JSONL}",
    "temperature": float("${TEMPERATURE}"),
    "reward_mode": "${REWARD_MODE}",
    "rl_mode": "existing_rollout_comp_valid_priority",
    "lr": float("${LR}"),
    "clip_eps": float("${CLIP_EPS}"),
    "beta": float("${BETA}"),
    "trace_shrink": int("${TRACE_SHRINK}"),
    "max_train_steps": int("${MAX_TRAIN_STEPS}"),
    "cepo_zero_non_credit": bool(int("${CEPO_ZERO_NON_CREDIT}")),
}, indent=2) + "\\n")
PY

python scripts/rl_reward_crystals.py \
  --input-jsonl "${RUN_DIR}/outputs/rollout/rollout_raw.jsonl" \
  --output-jsonl "${RUN_DIR}/outputs/rollout/rollout_scored.jsonl" \
  --summary-json "${RUN_DIR}/notes/reward_summary.json" \
  --summary-md "${RUN_DIR}/notes/reward_summary.md" \
  --reward-mode "${REWARD_MODE}"

python scripts/build_cepo_lite_evidence.py \
  --input-jsonl "${RUN_DIR}/outputs/rollout/rollout_scored.jsonl" \
  --output-jsonl "${RUN_DIR}/outputs/rollout/rollout_cepo.jsonl" \
  --summary-json "${RUN_DIR}/notes/cepo_summary.json" \
  --lambda-weight "${CEPO_LAMBDA}" \
  --clip-eps "${CEPO_CLIP_EPS}" \
  --positive-all-comp-valid \
  $(if [ "${CEPO_ZERO_NON_CREDIT}" = "1" ]; then printf -- "--zero-non-credit"; fi)

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
reward = json.loads((run / "notes/reward_summary.json").read_text())
cepo = json.loads((run / "notes/cepo_summary.json").read_text())
gate = {
    "reward_std": reward.get("reward_std"),
    "smact_valid_rate": reward.get("smact_valid_rate"),
    "charge_neutral_pauling_valid_rate": reward.get("charge_neutral_pauling_valid_rate"),
    "reason_counts": reward.get("reason_counts"),
    "group_has_pos_neg_rate": cepo.get("group_has_pos_neg_rate"),
    "credit_clip_ratio": cepo.get("credit_clip_ratio"),
}
print("RL pretrain gate", json.dumps(gate, ensure_ascii=False, indent=2))
(run / "notes" / "pretrain_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\\n")
if float(reward.get("reward_std") or 0.0) < 1e-6:
    raise SystemExit("reward std is zero; stop before TraceRL")
PY

torchrun --nproc_per_node=2 scripts/llada_trace_rl.py \
  --checkpoint-path "${BEST_CHECKPOINT}" \
  --rollout-jsonl "${RUN_DIR}/outputs/rollout/rollout_cepo.jsonl" \
  --output-dir "${RUN_DIR}/outputs/tracerl" \
  --lr "${LR}" \
  --clip-eps "${CLIP_EPS}" \
  --beta "${BETA}" \
  --trace-shrink "${TRACE_SHRINK}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --logging-steps 5 \
  --save-steps "${MAX_TRAIN_STEPS}"

torchrun --nproc_per_node=2 scripts/sample_llada_crystals.py \
  --checkpoint-path "${RUN_DIR}/outputs/tracerl/final" \
  --output-dir "${RUN_DIR}/outputs/sample256" \
  --num-samples "${SAMPLE_COUNT}" \
  --batch-size "${SAMPLE_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}"

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --failure-jsonl "${RUN_DIR}/outputs/sample256/failure_cases.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_distribution.json" \
  --output-md "${RUN_DIR}/notes/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_composition.json" \
  --output-md "${RUN_DIR}/notes/sample256_composition.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${RUN_DIR}/outputs/sample256/raw_generations.jsonl" \
  --output-json "${RUN_DIR}/notes/sample256_failure_modes.json" \
  --output-md "${RUN_DIR}/notes/sample256_failure_modes.md"

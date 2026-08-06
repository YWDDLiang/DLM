#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the A800 tmux session (ssha800), not from a direct SSH shell.

RUN_ID="${RUN_ID:-20260528_stcompress_abl0_r2_baseline}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
if [ ! -d "${PROJECT_ROOT}" ] && [ -d "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion" ]; then
  PROJECT_ROOT="/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"
fi
ENV_NAME="${ENV_NAME:-diff_meets_diff}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
R2_CKPT="${R2_CKPT:-runs/20260527_semalign_selfimprove_r2/outputs/stage_b/final}"
LOWLR5_CKPT="${LOWLR5_CKPT:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
LEGACY_PROJECT_ROOT="${LEGACY_PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GPU_COUNT="${GPU_COUNT:-2}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_MEM="${SLURM_MEM:-160G}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <= 2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}

Purpose:
- ABL-0 full-token R2 baseline resmoke for special-token compression ablation.

Protocol:
- fixed_slot, temperature=${TEMPERATURE}, block_length=1, n-elements-sequential-rest.
- schema mask, slot prefill, atom-count grammar, duplicate-coordinate mask, lattice-volume mask enabled.
- 256 raw samples only; no training.
EOF

DRIVER="runs/${RUN_ID}/slurm/abl0_driver.sh"
mkdir -p "runs/${RUN_ID}/slurm"
cat > "${DRIVER}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
R2_CKPT=$(printf '%q' "${R2_CKPT}")
LOWLR5_CKPT=$(printf '%q' "${LOWLR5_CKPT}")
MODEL_PATH=$(printf '%q' "${MODEL_PATH}")
TEMPERATURE=$(printf '%q' "${TEMPERATURE}")
GPU_COUNT=$(printf '%q' "${GPU_COUNT}")
RUN_ID=$(printf '%q' "${RUN_ID}")
PROJECT_ROOT=$(printf '%q' "${PROJECT_ROOT}")
LEGACY_PROJECT_ROOT=$(printf '%q' "${LEGACY_PROJECT_ROOT}")

EOF
cat >> "${DRIVER}" <<'EOF'
resolve_checkpoint() {
  local candidate="$1"
  if [ -d "${candidate}" ]; then
    printf '%s\n' "${candidate}"
  elif [ -d "${PROJECT_ROOT}/${candidate}" ]; then
    printf '%s\n' "${PROJECT_ROOT}/${candidate}"
  elif [ -d "${LEGACY_PROJECT_ROOT}/${candidate}" ]; then
    printf '%s\n' "${LEGACY_PROJECT_ROOT}/${candidate}"
  else
    printf '%s\n' "${candidate}"
  fi
}

R2_CKPT="$(resolve_checkpoint "${R2_CKPT}")"
LOWLR5_CKPT="$(resolve_checkpoint "${LOWLR5_CKPT}")"
CKPT="${R2_CKPT}"
if [ ! -d "${CKPT}" ] || [ ! -f "${CKPT}/adapter_config.json" ]; then
  echo "R2 checkpoint not found at ${CKPT}; falling back to ${LOWLR5_CKPT}"
  CKPT="${LOWLR5_CKPT}"
fi
if [ ! -d "${CKPT}" ]; then
  echo "No baseline checkpoint found: ${CKPT}" >&2
  exit 2
fi

SAMPLE_DIR="runs/${RUN_ID}/outputs/sample256"
NOTES_DIR="runs/${RUN_ID}/notes"
mkdir -p "${SAMPLE_DIR}" "${NOTES_DIR}"

python - <<PY
import json
from pathlib import Path
Path("${NOTES_DIR}/abl0_config.json").write_text(json.dumps({
  "checkpoint": "${CKPT}",
  "model_path": "${MODEL_PATH}",
  "temperature": float("${TEMPERATURE}"),
  "representation": "fixed_slot",
  "sample_count": 256,
}, indent=2) + "\n")
PY

torchrun --nproc_per_node="${GPU_COUNT}" scripts/sample_llada_crystals.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CKPT}" \
  --representation fixed_slot \
  --output-dir "${SAMPLE_DIR}" \
  --num-samples 256 \
  --batch-size 8 \
  --temperature "${TEMPERATURE}" \
  --block-length 1 \
  --generation-schedule n-elements-sequential-rest

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --representation fixed_slot \
  --output-json "${NOTES_DIR}/sample256_distribution.json" \
  --output-md "${NOTES_DIR}/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --representation fixed_slot \
  --output-json "${NOTES_DIR}/sample256_composition.json" \
  --output-md "${NOTES_DIR}/sample256_composition.md"
EOF
chmod +x "${DRIVER}"

R2_CKPT="${R2_CKPT}" \
LOWLR5_CKPT="${LOWLR5_CKPT}" \
MODEL_PATH="${MODEL_PATH}" \
TEMPERATURE="${TEMPERATURE}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
ENV_NAME="${ENV_NAME}" \
GPU_COUNT="${GPU_COUNT}" \
SLURM_TIME="${SLURM_TIME}" \
SLURM_MEM="${SLURM_MEM}" \
JOB_NAME="stc-abl0" \
scripts/a800/slurm_submit.sh "${RUN_ID}" bash "${DRIVER}"

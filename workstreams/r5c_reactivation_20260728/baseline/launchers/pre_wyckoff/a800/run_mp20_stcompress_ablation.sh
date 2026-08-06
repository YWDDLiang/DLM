#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the A800 tmux session (ssha800). This submits a Slurm job.

VARIANT="${VARIANT:-abl1}"  # abl1, abl2, abl3
case "${VARIANT}" in
  abl1|abl2|abl3) ;;
  *) echo "VARIANT must be abl1/abl2/abl3" >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-20260528_stcompress_${VARIANT}}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
if [ ! -d "${PROJECT_ROOT}" ] && [ -d "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion" ]; then
  PROJECT_ROOT="/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"
fi
ENV_NAME="${ENV_NAME:-diff_meets_diff}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
SOURCE_CKPT="${SOURCE_CKPT:-runs/20260527_semalign_selfimprove_r2/outputs/stage_b/final}"
FALLBACK_CKPT="${FALLBACK_CKPT:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
LEGACY_PROJECT_ROOT="${LEGACY_PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
TEMPERATURE="${TEMPERATURE:-0.7}"
LR="${LR:-1e-6}"
LIMIT_TRAIN="${LIMIT_TRAIN:-16000}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/dlm_sft/mp_20}"
GPU_COUNT="${GPU_COUNT:-2}"
SLURM_TIME="${SLURM_TIME:-10:00:00}"
SLURM_MEM="${SLURM_MEM:-220G}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <= 2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}

Purpose:
- ${VARIANT} special-token compression ablation.

Variant:
- abl1: shared XYZ as C tokens.
- abl2: shared XYZ + shared lattice length.
- abl3: shared XYZ + shared lattice length + shared angle.

Protocol:
- Convert checkpoint embeddings/head by averaging source token rows.
- 32-row SFT smoke + 64 sample parser/graph smoke.
- Short SFT: one capped epoch using LIMIT_TRAIN=${LIMIT_TRAIN}, lr=${LR}.
- 256 raw smoke at temperature=${TEMPERATURE}.
EOF

DRIVER="runs/${RUN_ID}/slurm/${VARIANT}_driver.sh"
mkdir -p "runs/${RUN_ID}/slurm"
cat > "${DRIVER}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
VARIANT=$(printf '%q' "${VARIANT}")
SOURCE_CKPT=$(printf '%q' "${SOURCE_CKPT}")
FALLBACK_CKPT=$(printf '%q' "${FALLBACK_CKPT}")
MODEL_PATH=$(printf '%q' "${MODEL_PATH}")
TEMPERATURE=$(printf '%q' "${TEMPERATURE}")
LR=$(printf '%q' "${LR}")
LIMIT_TRAIN=$(printf '%q' "${LIMIT_TRAIN}")
BASE_DATA_DIR=$(printf '%q' "${BASE_DATA_DIR}")
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

resolve_dir() {
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

SOURCE_CKPT="$(resolve_checkpoint "${SOURCE_CKPT}")"
FALLBACK_CKPT="$(resolve_checkpoint "${FALLBACK_CKPT}")"
BASE_DATA_DIR="$(resolve_dir "${BASE_DATA_DIR}")"
DATA_DIR="runs/${RUN_ID}/outputs/data_${VARIANT}"
CONVERTED_CKPT="runs/${RUN_ID}/outputs/converted_${VARIANT}"
SMOKE_DIR="runs/${RUN_ID}/outputs/smoke_sft"
SMOKE_SAMPLE_DIR="runs/${RUN_ID}/outputs/smoke_sample64"
SFT_DIR="runs/${RUN_ID}/outputs/sft_${VARIANT}"
SAMPLE_DIR="runs/${RUN_ID}/outputs/sample256"
NOTES_DIR="runs/${RUN_ID}/notes"
mkdir -p "${NOTES_DIR}" "runs/${RUN_ID}/outputs"

SRC="${SOURCE_CKPT}"
if [ ! -d "${SRC}" ] || [ ! -f "${SRC}/adapter_config.json" ]; then
  echo "Source checkpoint not found at ${SRC}; falling back to ${FALLBACK_CKPT}"
  SRC="${FALLBACK_CKPT}"
fi
if [ ! -d "${SRC}" ] || [ ! -f "${SRC}/adapter_config.json" ]; then
  echo "No valid source checkpoint found: ${SRC}" >&2
  exit 2
fi

python scripts/build_fixed_slot_compressed_sft_data.py \
  --input-dir "${BASE_DATA_DIR}" \
  --output-dir "${DATA_DIR}" \
  --tokenizer-path "${MODEL_PATH}" \
  --variant "${VARIANT}" \
  --answer-separator "" \
  --skip-graph-validation

python scripts/convert_fixed_slot_checkpoint_to_compressed.py \
  --source-checkpoint "${SRC}" \
  --output-dir "${CONVERTED_CKPT}" \
  --base-model-path "${MODEL_PATH}" \
  --variant "${VARIANT}"

torchrun --nproc_per_node="${GPU_COUNT}" scripts/llada_sft.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CONVERTED_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --representation fixed_slot_compressed_v1 \
  --output-dir "${SMOKE_DIR}" \
  --epochs 1 \
  --limit-train 32 \
  --limit-val 32 \
  --batch-size 4 \
  --grad-accum 1 \
  --lr "${LR}" \
  --lr-scheduler cosine \
  --warmup-steps 2 \
  --max-length 256 \
  --train-prefill-slot-tokens \
  --atom-count-loss-weight 3.0 \
  --slot-marker-loss-weight 0.25 \
  --empty-slot-loss-weight 0.2 \
  --nonempty-slot-loss-weight 2.0 \
  --late-nonempty-slot-loss-weight 4.0 \
  --coordinate-loss-weight 1.0 \
  --pad-coordinate-loss-weight 0.1

torchrun --nproc_per_node="${GPU_COUNT}" scripts/sample_llada_crystals.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${SMOKE_DIR}/final" \
  --representation fixed_slot_compressed_v1 \
  --compressed-token-config "${DATA_DIR}/compressed_token_config.json" \
  --output-dir "${SMOKE_SAMPLE_DIR}" \
  --num-samples 64 \
  --batch-size 8 \
  --temperature "${TEMPERATURE}" \
  --block-length 1 \
  --generation-schedule n-elements-sequential-rest

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SMOKE_SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SMOKE_SAMPLE_DIR}/failure_cases.jsonl" \
  --representation fixed_slot_compressed_v1 \
  --compressed-token-config "${DATA_DIR}/compressed_token_config.json" \
  --output-json "${NOTES_DIR}/smoke64_distribution.json" \
  --output-md "${NOTES_DIR}/smoke64_distribution.md"

torchrun --nproc_per_node="${GPU_COUNT}" scripts/llada_sft.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CONVERTED_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --representation fixed_slot_compressed_v1 \
  --output-dir "${SFT_DIR}" \
  --epochs 1 \
  --limit-train "${LIMIT_TRAIN}" \
  --limit-val 512 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr "${LR}" \
  --lr-scheduler cosine \
  --warmup-steps 50 \
  --max-length 256 \
  --logging-steps 20 \
  --eval-steps 250 \
  --save-steps 1000 \
  --position-diagnostics-steps 250 \
  --train-prefill-slot-tokens \
  --atom-count-loss-weight 3.0 \
  --slot-marker-loss-weight 0.25 \
  --empty-slot-loss-weight 0.2 \
  --nonempty-slot-loss-weight 2.0 \
  --late-nonempty-slot-loss-weight 4.0 \
  --coordinate-loss-weight 1.0 \
  --pad-coordinate-loss-weight 0.1

find "${SFT_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort | head -n -1 | xargs -r rm -rf

torchrun --nproc_per_node="${GPU_COUNT}" scripts/sample_llada_crystals.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${SFT_DIR}/final" \
  --representation fixed_slot_compressed_v1 \
  --compressed-token-config "${DATA_DIR}/compressed_token_config.json" \
  --output-dir "${SAMPLE_DIR}" \
  --num-samples 256 \
  --batch-size 8 \
  --temperature "${TEMPERATURE}" \
  --block-length 1 \
  --generation-schedule n-elements-sequential-rest

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE_DIR}/failure_cases.jsonl" \
  --representation fixed_slot_compressed_v1 \
  --compressed-token-config "${DATA_DIR}/compressed_token_config.json" \
  --output-json "${NOTES_DIR}/sample256_distribution.json" \
  --output-md "${NOTES_DIR}/sample256_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE_DIR}/raw_generations.jsonl" \
  --representation fixed_slot_compressed_v1 \
  --compressed-token-config "${DATA_DIR}/compressed_token_config.json" \
  --output-json "${NOTES_DIR}/sample256_composition.json" \
  --output-md "${NOTES_DIR}/sample256_composition.md"

python scripts/evaluate_mp20_candidate_gate.py \
  --mode smoke256 \
  --sample-metrics "${SAMPLE_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/sample256_composition.json" \
  --composition-key raw_jsonl \
  --min-graph-acceptance 0.95 \
  --min-comp-valid 0.87 \
  --min-strict-valid 0.40 \
  --max-single-element 0.10 \
  --max-pbc-duplicate 0.0 \
  --output-json "${NOTES_DIR}/sample256_gate.json" || true
EOF
chmod +x "${DRIVER}"

VARIANT="${VARIANT}" \
SOURCE_CKPT="${SOURCE_CKPT}" \
FALLBACK_CKPT="${FALLBACK_CKPT}" \
MODEL_PATH="${MODEL_PATH}" \
TEMPERATURE="${TEMPERATURE}" \
LR="${LR}" \
LIMIT_TRAIN="${LIMIT_TRAIN}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
ENV_NAME="${ENV_NAME}" \
GPU_COUNT="${GPU_COUNT}" \
SLURM_TIME="${SLURM_TIME}" \
SLURM_MEM="${SLURM_MEM}" \
JOB_NAME="stc-${VARIANT}" \
scripts/a800/slurm_submit.sh "${RUN_ID}" bash "${DRIVER}"

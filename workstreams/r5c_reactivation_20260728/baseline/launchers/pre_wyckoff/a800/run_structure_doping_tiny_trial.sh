#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-structure-doping-tiny-trial}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
MP20_ADAPTER="${MP20_ADAPTER:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
STRUCT20_CKPT="${STRUCT20_CKPT:-runs/20260520_002000-structure-doping-fast-trial/outputs/structure20_sft/final}"

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes" "runs/${RUN_ID}/outputs"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}

Tiny validation run:

- structure20: sample 8 examples from an existing structure20 adapter when available.
- full80: train 32-row / 1-epoch tiny direct full80 baseline and sample 4 examples.
- This is only a pipeline and initial capability probe, not a statistically sufficient proof.
EOF

python scripts/build_doping_structure20_data.py --output-dir data/doping_structure20
python scripts/build_doping_full80_holdout_data.py --output-dir data/doping_full80_holdout

if [ ! -d "${STRUCT20_CKPT}" ]; then
  if [ ! -d "${MP20_ADAPTER}" ]; then
    echo "Missing structure20 checkpoint and MP-20 adapter." >&2
    exit 3
  fi
  STRUCT20_CKPT="runs/${RUN_ID}/outputs/structure20_sft/final"
  torchrun --standalone --nproc_per_node=2 scripts/llada_sft.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${MP20_ADAPTER}" \
    --data-dir data/doping_structure20 \
    --output-dir "runs/${RUN_ID}/outputs/structure20_sft" \
    --max-length 192 \
    --answer-token-count 107 \
    --skip-data-vocab-resize \
    --save-embedding-layers false \
    --epochs 1 \
    --limit-train 32 \
    --limit-val 8 \
    --batch-size 4 \
    --grad-accum 2 \
    --lr 5e-6 \
    --lr-scheduler cosine \
    --warmup-steps 2 \
    --logging-steps 2 \
    --eval-steps 8 \
    --eval-max-batches 2 \
    --save-steps 100
fi

python scripts/sample_doping_structure20.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${STRUCT20_CKPT}" \
  --data-dir data/doping_structure20 \
  --output-dir "runs/${RUN_ID}/outputs/structure20_sample" \
  --num-samples 8 \
  --batch-size 4 \
  --steps 16 \
  --temperature 0.0

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "runs/${RUN_ID}/outputs/structure20_sample" \
  --data-dir data/doping_structure20

torchrun --standalone --nproc_per_node=2 scripts/llada_sft.py \
  --model-path "${MODEL_PATH}" \
  --data-dir data/doping_full80_holdout \
  --output-dir "runs/${RUN_ID}/outputs/full80_sft" \
  --max-length 512 \
  --answer-token-count 407 \
  --epochs 1 \
  --limit-train 32 \
  --limit-val 8 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 2 \
  --logging-steps 2 \
  --eval-steps 8 \
  --eval-max-batches 2 \
  --save-steps 100

python scripts/sample_doping_full80.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "runs/${RUN_ID}/outputs/full80_sft/final" \
  --data-dir data/doping_full80_holdout \
  --output-dir "runs/${RUN_ID}/outputs/full80_sample" \
  --num-samples 4 \
  --batch-size 2 \
  --steps 32 \
  --temperature 0.0

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "runs/${RUN_ID}/outputs/full80_sample" \
  --data-dir data/doping_full80_holdout

python scripts/write_structure_doping_trial_report.py \
  --run-dir "runs/${RUN_ID}" \
  --output-report "reports/20260520_structure20_vs_full80_trial_report.md"

cat > "runs/${RUN_ID}/notes/ai_review.md" <<EOF
# AI Review

本轮是 tiny validation，不是充分统计实验。

请阅读：

- runs/${RUN_ID}/outputs/structure20_sample/structure_similarity_eval.md
- runs/${RUN_ID}/outputs/full80_sample/structure_similarity_eval.md
- reports/20260520_structure20_vs_full80_trial_report.md

没有 DFT relaxation，因此不能声称“性质更好”。
EOF

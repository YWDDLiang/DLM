#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-structure20-inline-model}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
MP20_ADAPTER="${MP20_ADAPTER:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes" "runs/${RUN_ID}/outputs"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}
EOF
cat >> "runs/${RUN_ID}/notes/plan.md" <<'EOF'

本轮是 structure20 inline model trial：

- 不保存/重载新的 PEFT adapter，避免 6GB adapter 阻塞。
- 从 MP-20 adapter 继续 SFT，使用完整 `data/doping_structure20/train.jsonl` oversampled split。
- 训练 5 epoch 后由模型自己生成 dopants，再生成 structure20 geometry，展开为完整 80-atom CIF。
- 采样 32 条并运行 graph build 与 hidden-good similarity 评估。

结论边界：没有 DFT relaxation，不能声称“性质更好”；若 graph/similarity 过关，只能声称 verified-good-like 结构生成证据增强。
EOF

if [ ! -d "${MP20_ADAPTER}" ]; then
  echo "Missing MP-20 adapter: ${MP20_ADAPTER}" | tee "runs/${RUN_ID}/notes/ai_review.md"
  exit 3
fi

python scripts/build_doping_structure20_data.py --output-dir data/doping_structure20

timeout 90m python scripts/structure20_inline_tiny_trial.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${MP20_ADAPTER}" \
  --data-dir data/doping_structure20 \
  --output-dir "runs/${RUN_ID}/outputs/structure20_inline_model_32_graph" \
  --epochs 5 \
  --limit-train 309 \
  --train-batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --num-samples 32 \
  --sample-batch-size 1 \
  --steps 32 \
  --temperature 0.8 \
  --dopant-mode model

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "runs/${RUN_ID}/outputs/structure20_inline_model_32_graph" \
  --data-dir data/doping_structure20

cat > "runs/${RUN_ID}/notes/ai_review.md" <<'EOF'
# AI Review

本轮中文结论待根据 metrics 判断：

- 重点看 `sample_metrics.json` 的 `graph_build_rate` 和 `unique_expanded_structure_count`。
- 重点看 `structure_similarity_eval.md` 是否出现 hidden-good same-element / near-hit。
- 如果 graph rate 仍低，优先分析 `raw_generations.jsonl` 中重复 B-site coordinate 和 lattice tail，而不是继续盲目加采样。
- 没有 DFT relaxation，因此不能写“模型已经证明生成了性质更好的晶体结构”。
EOF

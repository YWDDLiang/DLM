#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-structure20-inline-tiny}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
MP20_ADAPTER="${MP20_ADAPTER:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes" "runs/${RUN_ID}/outputs"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}
EOF
cat >> "runs/${RUN_ID}/notes/plan.md" <<'EOF'

本轮是 structure20 inline tiny debug：

- 清理失败 run 后，先避免再次保存/加载 6GB PEFT adapter。
- 在同一个 Python 进程中从 MP-20 adapter 继续 tiny SFT，然后立即采样。
- 第一段固定 dopants=Ca,Fe,Ni，采样 1 条，steps=4，skip graph，用于确认 model load、train、first generate 是否跑通。
- 第一段成功后再固定同一 dopants 采样 8 条，steps=8，启用 graph build，检查完整结构输出链路。
- 采样阶段保留 train mode，用于诊断 LLaDA remote code 是否存在 eval-mode forward hang。

结论边界：没有 DFT relaxation，不能声称“性质更好”。
EOF

if [ ! -d "${MP20_ADAPTER}" ]; then
  echo "Missing MP-20 adapter: ${MP20_ADAPTER}" | tee "runs/${RUN_ID}/notes/ai_review.md"
  exit 3
fi

python scripts/build_doping_structure20_data.py --output-dir data/doping_structure20

timeout 20m python scripts/structure20_inline_tiny_trial.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${MP20_ADAPTER}" \
  --data-dir data/doping_structure20 \
  --output-dir "runs/${RUN_ID}/outputs/structure20_inline_1_skipgraph" \
  --epochs 1 \
  --limit-train 16 \
  --train-batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --num-samples 1 \
  --sample-batch-size 1 \
  --steps 4 \
  --temperature 0.0 \
  --dopant-mode fixed \
  --fixed-dopants Ca,Fe,Ni \
  --sample-train-mode \
  --skip-graph

timeout 30m python scripts/structure20_inline_tiny_trial.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${MP20_ADAPTER}" \
  --data-dir data/doping_structure20 \
  --output-dir "runs/${RUN_ID}/outputs/structure20_inline_8_graph" \
  --epochs 1 \
  --limit-train 32 \
  --train-batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --num-samples 8 \
  --sample-batch-size 1 \
  --steps 8 \
  --temperature 0.0 \
  --dopant-mode fixed \
  --fixed-dopants Ca,Fe,Ni \
  --sample-train-mode

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "runs/${RUN_ID}/outputs/structure20_inline_8_graph" \
  --data-dir data/doping_structure20

cat > "runs/${RUN_ID}/notes/ai_review.md" <<'EOF'
# AI Review

本轮中文结论：

- 这是 inline tiny debug，用来区分“adapter 保存/加载问题”和“生成/解析/graph build 问题”。
- 首先查看 progress.jsonl，确认是否卡在 model_load、train、generate 或 graph_build。
- 若 1-sample skip-graph 和 8-sample graph 都通过，下一步才恢复正式 adapter 保存/采样，或改成双阶段加载 MP-20 adapter + structure20 delta adapter。
- 没有 DFT relaxation，因此不能声称“模型已经证明生成了性质更好的晶体结构”。
EOF

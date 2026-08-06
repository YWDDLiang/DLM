#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-structure-doping-dual-trial}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
MP20_ADAPTER="${MP20_ADAPTER:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/notes" "runs/${RUN_ID}/outputs"

cat > "runs/${RUN_ID}/notes/plan.md" <<EOF
# ${RUN_ID}

本轮执行 structure-aware doping 双实验：

- 实验 A：DOPING_STRUCT20 compressed structural code，展开为完整 80 原子 CIF。
- 实验 B：DOPING_FULL80 direct 407-token fixed-slot structure。
- 对比 baseline：compact-template good-holdout sampling。
- 结论边界：没有 DFT relaxation 时，只能声称 verified-good-like structure similarity，不能声称性质已经更好。
EOF

if [ ! -d "${MP20_ADAPTER}" ]; then
  echo "Missing MP-20 adapter: ${MP20_ADAPTER}" | tee "runs/${RUN_ID}/notes/ai_review.md"
  exit 3
fi

python scripts/build_doping_structure20_data.py --output-dir data/doping_structure20
python scripts/build_doping_full80_holdout_data.py --output-dir data/doping_full80_holdout

STRUCT20_SFT="runs/${RUN_ID}/outputs/structure20_sft"
FULL80_SFT="runs/${RUN_ID}/outputs/full80_sft"
STRUCT20_SAMPLE="runs/${RUN_ID}/outputs/structure20_sample"
FULL80_SAMPLE="runs/${RUN_ID}/outputs/full80_sample"

torchrun --standalone --nproc_per_node=2 scripts/llada_sft.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${MP20_ADAPTER}" \
  --data-dir data/doping_structure20 \
  --output-dir "${STRUCT20_SFT}" \
  --max-length 192 \
  --answer-token-count 107 \
  --skip-data-vocab-resize \
  --save-embedding-layers false \
  --epochs 10 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 20 \
  --min-lr-ratio 0.1 \
  --logging-steps 10 \
  --eval-steps 50 \
  --eval-max-batches 4 \
  --save-steps 200

python scripts/sample_doping_structure20.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${STRUCT20_SFT}/final" \
  --data-dir data/doping_structure20 \
  --output-dir "${STRUCT20_SAMPLE}" \
  --num-samples 128 \
  --batch-size 8 \
  --steps 32 \
  --temperature 1.0

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "${STRUCT20_SAMPLE}" \
  --data-dir data/doping_structure20

set +e
RUN_ID="${RUN_ID}" python - <<'PY'
import json
import os
from pathlib import Path
run_id = os.environ["RUN_ID"]
metrics = json.loads(Path(f"runs/{run_id}/outputs/structure20_sample/sample_metrics.json").read_text())
ok = (
    metrics.get("parse_rate", 0) >= 0.95
    and metrics.get("reconstruction_rate", 0) >= 0.95
    and metrics.get("composition_exact_rate", 0) >= 0.95
    and (metrics.get("graph_build_rate") is None or metrics.get("graph_build_rate", 0) >= 0.85)
)
Path(f"runs/{run_id}/notes/structure20_gate.json").write_text(json.dumps({"pass": ok, "metrics": metrics}, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(0 if ok else 10)
PY
STRUCT20_GATE_STATUS=$?
set -e
if [ "${STRUCT20_GATE_STATUS}" -eq 0 ] && [ "${RUN_FORMAL:-0}" = "1" ]; then
  python scripts/sample_doping_structure20.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${STRUCT20_SFT}/final" \
    --data-dir data/doping_structure20 \
    --output-dir "runs/${RUN_ID}/outputs/structure20_sample_2048" \
    --num-samples 512 \
    --batch-size 8 \
    --steps 32 \
    --temperature 1.0
  python scripts/evaluate_doping_structure_similarity.py \
    --sample-output-dir "runs/${RUN_ID}/outputs/structure20_sample_2048" \
    --data-dir data/doping_structure20
fi

torchrun --standalone --nproc_per_node=2 scripts/llada_sft.py \
  --model-path "${MODEL_PATH}" \
  --data-dir data/doping_full80_holdout \
  --output-dir "${FULL80_SFT}" \
  --max-length 512 \
  --answer-token-count 407 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 20 \
  --min-lr-ratio 0.1 \
  --logging-steps 10 \
  --eval-steps 50 \
  --eval-max-batches 4 \
  --save-steps 200

python scripts/sample_doping_full80.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${FULL80_SFT}/final" \
  --data-dir data/doping_full80_holdout \
  --output-dir "${FULL80_SAMPLE}" \
  --num-samples 32 \
  --batch-size 2 \
  --steps 64 \
  --temperature 1.0

python scripts/evaluate_doping_structure_similarity.py \
  --sample-output-dir "${FULL80_SAMPLE}" \
  --data-dir data/doping_full80_holdout

python scripts/write_structure_doping_trial_report.py \
  --run-dir "runs/${RUN_ID}" \
  --output-report "reports/20260520_structure20_vs_full80_trial_report.md"

cat > "runs/${RUN_ID}/notes/ai_review.md" <<EOF
# AI Review

本轮已完成 structure20 与 full80 的离线结构验证尝试。请优先阅读：

- reports/20260520_structure20_vs_full80_trial_report.md
- runs/${RUN_ID}/outputs/structure20_sample/structure_similarity_eval.md
- runs/${RUN_ID}/outputs/full80_sample/structure_similarity_eval.md

若没有 DFT relaxation，本轮不能写“性质已经更好”，只能写 verified-good-like structure similarity。
EOF

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-story-e1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-story-e1-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly LEARNED_PLANS="${STORY_LEARNED_PLANS:-${REPO_ROOT}/data/plans/h1a2_learned_rich.jsonl}"
readonly GOLD_PLANS="${STORY_GOLD_PLANS:-${REPO_ROOT}/data/plans/r5c_gold_rich.jsonl}"
readonly E1_DIR="${REPO_ROOT}/runs/story_e1"
readonly CONTRACT_DIR="${E1_DIR}/contracts"
readonly OUTPUT_DIR="${E1_DIR}/body"
readonly MODEL_BASE="${REPO_ROOT}/checkpoints/dlm/base"
readonly MODEL_ADAPTER="${REPO_ROOT}/checkpoints/dlm"
readonly CRYSLLMGEN_DIR="${REPO_ROOT}/src/vendor/crysllmgen"

[[ -f "${LEARNED_PLANS}" ]] || missing "${LEARNED_PLANS}"
[[ -f "${GOLD_PLANS}" ]] || missing "${GOLD_PLANS}"
[[ -d "${MODEL_BASE}" ]] || missing "${MODEL_BASE}"
[[ -f "${MODEL_ADAPTER}/adapter_model.safetensors" ]] || missing "${MODEL_ADAPTER}/adapter_model.safetensors"

python "${REPO_ROOT}/scripts/build_story_panels.py" \
  --learned-plans "${LEARNED_PLANS}" \
  --gold-plans "${GOLD_PLANS}" \
  --output-dir "${CONTRACT_DIR}" \
  --num-pairs 24 --e2-pairs 8 --seed 20260822

python "${REPO_ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
  --model-path "${MODEL_BASE}" \
  --checkpoint-path "${MODEL_ADAPTER}" \
  --output-dir "${OUTPUT_DIR}" \
  --crysllmgen-dir "${CRYSLLMGEN_DIR}" \
  --task-ledger "${CONTRACT_DIR}/e1_tasks.jsonl" \
  --num-samples 768 --batch-size 1 --temperature 0.7 \
  --body-prompt-style full_plan_state \
  --schema-logit-mask --prefill-count-token --freeze-plan-composition \
  --duplicate-coordinate-mask --lattice-legality-mask \
  --generation-schedule exact-plan

python "${REPO_ROOT}/scripts/analyze_story_panel.py" \
  --tasks "${CONTRACT_DIR}/e1_tasks.jsonl" \
  --raw "${OUTPUT_DIR}/raw_generations.jsonl" \
  --sample-metrics "${OUTPUT_DIR}/sample_metrics.json" \
  --output-json "${E1_DIR}/e1_analysis.json" \
  --output-jsonl "${E1_DIR}/e1_records.jsonl"

log "E1 complete: ${OUTPUT_DIR}"

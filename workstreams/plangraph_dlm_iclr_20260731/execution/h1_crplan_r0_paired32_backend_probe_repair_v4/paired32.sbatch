#!/bin/bash
#SBATCH --job-name=h1-crplan-p32-v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_backend_probe_repair_v4/logs/%j_paired32.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_backend_probe_repair_v4/logs/%j_paired32.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260804_h1_crplan_r0_paired32_backend_probe_repair_v4"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_r0_paired32_backend_probe_repair_v4"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
HISTORICAL_P0_RAW="${PROJECT_ROOT}/runs/20260801_h1a2_v3_planner512_v1/arms/P0/plan512/raw_generations.jsonl"
HISTORICAL_P0_SHA256=7c630cecd7654d103f67ad2edf69da869c22408cec95454c26f2ddd6870d3e70
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing source manifest identity}"
: "${EXPECTED_ADAPTER_SHA256:?missing adapter identity}"

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test -d "${SOURCE_ROOT}"
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test -f "${HISTORICAL_P0_RAW}"
test "$(sha256sum "${HISTORICAL_P0_RAW}" | cut -d' ' -f1)" = "${HISTORICAL_P0_SHA256}"
test -f "${RUN_ROOT}/cr0/_SUCCESS"
test -f "${RUN_ROOT}/cr0/terminal_report.json"
test "$(sha256sum "${CHECKPOINT}/adapter_model.safetensors" | cut -d' ' -f1)" = "${EXPECTED_ADAPTER_SHA256}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test ! -e "${RUN_ROOT}/paired32"

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

"${PYTHON}" - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

mkdir -p "${RUN_ROOT}/paired32/control" "${RUN_ROOT}/paired32/candidate"

"${PYTHON}" scripts/sample_llama_h1_formula_plans.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/paired32/control" \
  --num-samples 32 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --seed 17029 \
  --seed-mode stateless_ordinal_v1 \
  --formula-constraint-mode off

"${PYTHON}" scripts/sample_llama_h1_formula_plans.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/paired32/candidate" \
  --num-samples 32 \
  --batch-size 4 \
  --max-new-tokens 96 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --max-atoms 20 \
  --prompt-style h1_rich_plan_v1 \
  --no-include-sample-id \
  --seed 17029 \
  --seed-mode stateless_ordinal_v1 \
  --formula-constraint-mode full_prefix

"${PYTHON}" "${EXECUTION_DIR}/evaluate_paired32.py" \
  --control-root "${RUN_ROOT}/paired32/control" \
  --candidate-root "${RUN_ROOT}/paired32/candidate" \
  --cr0-report "${RUN_ROOT}/cr0/terminal_report.json" \
  --source-manifest "${SOURCE_ROOT}/SOURCE_SHA256.txt" \
  --historical-p0-raw "${HISTORICAL_P0_RAW}" \
  --historical-p0-sha256 "${HISTORICAL_P0_SHA256}" \
  --output "${RUN_ROOT}/paired32/terminal_report.json"

touch "${RUN_ROOT}/paired32/_SUCCESS"
echo "H1_CRPLAN_PAIRED32_COMPLETE"

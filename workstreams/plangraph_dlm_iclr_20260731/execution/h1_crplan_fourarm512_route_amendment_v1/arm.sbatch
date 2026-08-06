#!/bin/bash
#SBATCH --job-name=h1-cr-4a512-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1/logs/%A_%a_arm.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1/logs/%A_%a_arm.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260805_h1_crplan_fourarm512_route_amendment_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_fourarm512_route_amendment_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing source manifest identity}"
: "${EXPECTED_LEDGER_SHA256:?missing science ledger identity}"
: "${EXPECTED_ADAPTER_SHA256:?missing adapter identity}"
: "${SLURM_ARRAY_TASK_ID:?missing array task identity}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODE=off ;;
  1) MODE=grammar_only ;;
  2) MODE=terminal_only ;;
  3) MODE=full_prefix ;;
  *) echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test -d "${SOURCE_ROOT}"
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test -x "${PYTHON}"
test ! -e "${RUN_ROOT}/arms/${MODE}"
test "$(sha256sum "${CHECKPOINT}/adapter_model.safetensors" | cut -d' ' -f1)" = "${EXPECTED_ADAPTER_SHA256}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "$(sha256sum "${EXECUTION_DIR}/SCIENCE_LEDGER.json" | cut -d' ' -f1)" = "${EXPECTED_LEDGER_SHA256}"

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
print({"gpu": names, "torch": torch.__version__}, flush=True)
PY

mkdir -p "${RUN_ROOT}/status"
set +e
"${PYTHON}" "${EXECUTION_DIR}/run_fourarm512_arm.py" \
  --run-root "${RUN_ROOT}" \
  --config "${EXECUTION_DIR}/CONFIG.json" \
  --ledger "${EXECUTION_DIR}/SCIENCE_LEDGER.json" \
  --source-manifest "${SOURCE_ROOT}/SOURCE_SHA256.txt" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --expected-adapter-sha256 "${EXPECTED_ADAPTER_SHA256}" \
  --mode "${MODE}"
runner_rc=$?
set -e
printf '%s\n' "${runner_rc}" > "${RUN_ROOT}/status/${MODE}_exit_code.txt"

if test "${runner_rc}" -eq 0; then
  touch "${RUN_ROOT}/status/${MODE}_SUCCESS"
  echo "H1_CRPLAN_FOURARM512_ARM_COMPLETE mode=${MODE}"
  exit 0
fi
touch "${RUN_ROOT}/status/${MODE}_FAILED"
echo "H1_CRPLAN_FOURARM512_ARM_FAILED mode=${MODE}" >&2
exit "${runner_rc}"

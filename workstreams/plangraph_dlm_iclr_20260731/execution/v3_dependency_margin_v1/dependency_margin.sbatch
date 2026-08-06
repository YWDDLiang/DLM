#!/bin/bash
#SBATCH --job-name=v3-dep-margin
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_dependency_margin_v1/logs/%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_dependency_margin_v1/logs/%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_v3_dependency_margin_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dependency_margin_v1"
AUTHORIZATION="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/AUTHORIZATION_V3_PLANNER512_DEPENDENCY_SCREEN_20260801.json"
PROTOCOL="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/PROTOCOL_ADDENDUM_V3_PLANNER512_DEPENDENCY_MARGIN_20260801.json"
MANIFEST="${EXECUTION_DIR}/EXECUTION_MANIFEST.json"
TRAINING_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1"
TRAINING_TERMINAL="${TRAINING_ROOT}/terminal_report.json"
MODEL_PATH=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
DATA_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_sidecar_preflight_v1/data/h1a2_r5c_plangraph_sidecar_v2"
VALIDATION_JSONL="${DATA_ROOT}/val.jsonl"
DATA_MANIFEST="${DATA_ROOT}/manifest.json"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_AUTH_SHA256:?submission must export EXPECTED_AUTH_SHA256}"
: "${EXPECTED_PROTOCOL_SHA256:?submission must export EXPECTED_PROTOCOL_SHA256}"
: "${EXPECTED_MANIFEST_SHA256:?submission must export EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_TRAINING_TERMINAL_SHA256:?submission must export EXPECTED_TRAINING_TERMINAL_SHA256}"
: "${EXPECTED_DATA_MANIFEST_SHA256:?submission must export EXPECTED_DATA_MANIFEST_SHA256}"
: "${EXPECTED_VALIDATION_SHA256:?submission must export EXPECTED_VALIDATION_SHA256}"

if [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -lt 0 ] \
  || [ "${SLURM_ARRAY_TASK_ID:-99}" -gt 1 ]; then
  echo "registered dependency-screen Slurm identity changed" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=B1
    POLICY=d1
    CHECKPOINT="${TRAINING_ROOT}/arms/B1/output/final"
    CHECKPOINT_IDENTITY=ace1a0d1f935e32881773228de5fcd5e91a76ca6e261ea132d609b84293bea8c
    ;;
  1)
    ARM=B2
    POLICY=d2
    CHECKPOINT="${TRAINING_ROOT}/arms/B2/output/final"
    CHECKPOINT_IDENTITY=e3451c94822ca83d1fa1f91680e24880ce9ced022379865b76705554f31f8cb8
    ;;
  *)
    exit 2
    ;;
esac

ARM_ROOT="${RUN_ROOT}/arms/${ARM}"

test -d "${SOURCE_ROOT}"
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test -f "${CHECKPOINT}/adapter_model.safetensors"
test -f "${VALIDATION_JSONL}"
test -f "${DATA_MANIFEST}"
test -f "${TRAINING_TERMINAL}"
test ! -e "${ARM_ROOT}"
test "$(sha256sum "${AUTHORIZATION}" | cut -d' ' -f1)" = "${EXPECTED_AUTH_SHA256}"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = "${EXPECTED_PROTOCOL_SHA256}"
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_MANIFEST_SHA256}"
test "$(sha256sum "${TRAINING_TERMINAL}" | cut -d' ' -f1)" = "${EXPECTED_TRAINING_TERMINAL_SHA256}"
test "$(sha256sum "${DATA_MANIFEST}" | cut -d' ' -f1)" = "${EXPECTED_DATA_MANIFEST_SHA256}"
test "$(sha256sum "${VALIDATION_JSONL}" | cut -d' ' -f1)" = "${EXPECTED_VALIDATION_SHA256}"

"${PYTHON}" - "${TRAINING_TERMINAL}" "${ARM}" "${POLICY}" "${CHECKPOINT}" "${CHECKPOINT_IDENTITY}" <<'PY'
import json
from pathlib import Path
import sys

terminal_path, arm, policy, checkpoint, expected_sha = sys.argv[1:]
terminal = json.loads(Path(terminal_path).read_text(encoding="utf-8"))
if terminal.get("status") != "complete" or terminal.get("training_complete") is not True:
    raise SystemExit("DLM one-epoch terminal is not complete")
if terminal.get("B2_likelihood_gate_passed") is not True:
    raise SystemExit("registered DLM likelihood gate did not pass")
record = terminal.get("arms", {}).get(arm, {})
if record.get("status") != "complete" or int(record.get("optimizer_updates", -1)) != 1696:
    raise SystemExit(f"{arm} is not a complete one-epoch endpoint")
if record.get("policy") != policy or record.get("final_checkpoint") != checkpoint:
    raise SystemExit(f"{arm} endpoint identity changed")
inventory = {
    item.get("path"): item
    for item in record.get("final_inventory", [])
    if isinstance(item, dict)
}
adapter = inventory.get("adapter_model.safetensors", {})
if adapter.get("sha256") != expected_sha or int(adapter.get("bytes", -1)) != 6391016776:
    raise SystemExit(f"{arm} adapter inventory identity changed")
PY

cd "${SOURCE_ROOT}"
sha256sum -c "${EXECUTION_DIR}/SOURCE_SHA256.txt"

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
print({"gpu": names, "torch": torch.__version__})
PY

mkdir -p "${ARM_ROOT}"
"${PYTHON}" "${EXECUTION_DIR}/evaluate_dependency_margin.py" \
  --arm "${ARM}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_IDENTITY}" \
  --validation-jsonl "${VALIDATION_JSONL}" \
  --data-manifest "${DATA_MANIFEST}" \
  --execution-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --batch-size 2 \
  --output "${ARM_ROOT}/dependency_report.json"

echo "DEPENDENCY_MARGIN_ARM_COMPLETE ARM=${ARM} POLICY=${POLICY}"

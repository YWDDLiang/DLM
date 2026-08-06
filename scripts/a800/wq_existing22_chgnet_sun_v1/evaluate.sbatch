#!/bin/bash
#SBATCH --job-name=wq-existing22-sun-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
PLAN=configs/experiments/wyckoff_codiffusion/wq_existing22_chgnet_sun_v1.json
OUTPUT="$RUN/outputs/wq_existing22_chgnet_sun_v1"
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
DIFF_ENV=/public/home/jiaosz/miniconda3/envs/diff_meets_diff
CHGNET_RUNTIME="$DIFF_ENV/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"

: "${EVALUATION_PATCH_SHA256:?submission must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]]; then
  printf '%s\n' "existing-22 evaluation identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test -f "$PLAN"
test ! -e "$OUTPUT"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
if [[ "${CONDA_DEFAULT_ENV:-}" != diff_meets_diff ]] \
  || [[ "${CONDA_PREFIX:-}" != "$DIFF_ENV" ]] \
  || [[ "$(command -v python)" != "$DIFF_ENV/bin/python" ]]; then
  printf '%s\n' "diff_meets_diff activation contract failed" >&2
  exit 3
fi

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0
unset MP_API_KEY
unset PMG_MAPI_KEY

mkdir "$OUTPUT"
usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

printf '%s\n' "pipeline_stage=environment_smoke"
python - "$OUTPUT/environment_contract.json" "$EVALUATION_PATCH_SHA256" \
  "$CHGNET_RUNTIME" <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

import torch
import torch_scatter

output = Path(sys.argv[1])
execution_patch = sys.argv[2]
chgnet_checkpoint = Path(sys.argv[3])
expected_prefix = Path(
    "/public/home/jiaosz/miniconda3/envs/diff_meets_diff"
)
if (
    os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff"
    or Path(os.environ.get("CONDA_PREFIX", "")).resolve() != expected_prefix
    or Path(sys.executable).resolve()
    != (expected_prefix / "bin/python3.10").resolve()
    or torch.__version__ != "2.4.0+cu121"
    or torch.version.cuda != "12.1"
    or md.version("torch-scatter") != "2.1.2+pt24cu121"
    or md.version("chgnet") != "0.4.2"
    or torch.cuda.device_count() != 1
    or "A800" not in torch.cuda.get_device_name(0)
    or not chgnet_checkpoint.is_file()
    or os.environ.get("MP_API_KEY")
    or os.environ.get("PMG_MAPI_KEY")
):
    raise SystemExit("existing-22 diff_meets_diff environment smoke failed")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "schema": "wqcodiff_existing22_chgnet_sun_environment_contract_v1",
    "ok": True,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "execution_patch_sha256": execution_patch,
    "conda_environment": os.environ["CONDA_DEFAULT_ENV"],
    "conda_prefix": os.environ["CONDA_PREFIX"],
    "python_executable": sys.executable,
    "python_version": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_scatter": md.version("torch-scatter"),
    "torch_scatter_file": torch_scatter.__file__,
    "chgnet": md.version("chgnet"),
    "chgnet_runtime_checkpoint": str(chgnet_checkpoint.resolve()),
    "chgnet_runtime_checkpoint_sha256": sha(chgnet_checkpoint),
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_device_count": torch.cuda.device_count(),
    "cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
    "mp_api_key_present": False,
    "offline": True,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("WQ_EXISTING22_ENVIRONMENT_SMOKE=PASS")
print(json.dumps(payload, sort_keys=True))
PY

printf '%s\n' "pipeline_stage=all22_attempt_adapter"
python -m diagnostics.prepare_wq_existing22_chgnet_sun \
  --contract "$PLAN" \
  --output-dir "$OUTPUT"

printf '%s\n' "pipeline_stage=exact_chgnet_r5c_sun"
python scripts/a800/run_crysllmgen_a100_sun.py \
  --generation-jsonl "$OUTPUT/all22_attempts.jsonl" \
  --expected-attempts 22 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
  --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint "$CHGNET_RUNTIME" \
  --base-source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --output-dir "$OUTPUT/r5c_a100_sun"

printf '%s\n' "pipeline_stage=terminal_acceptance"
python -m diagnostics.accept_wq_existing22_chgnet_sun \
  --contract "$PLAN" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --gpu-csv "$usage_csv"

#!/bin/bash
#SBATCH --job-name=wq-chemstop-p64-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_llm_charge_stop_paired64_v1
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_llm_charge_stop_paired64_v1.json
CONTRACT_SHA256=fdcd6b2b2a897fd4cc912e4d5f0e3053626c2cc19a7f510d5a7c0beb6d060307
PROTOCOL=configs/experiments/wyckoff_codiffusion/protocol_v4.yaml
GATE="$RUN/outputs/gate_a_lock_v8.json"
LLAMA_ROOT=/public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1/epoch_03/adapter_final"
OUTPUT="$RUN/outputs/$IDENTITY"
RUNNER=scripts/a800/run_wq_llm_charge_stop_paired64_v1.py

: "${WQ_CHARGE_STOP_PATCH_SHA256:?submission must export WQ_CHARGE_STOP_PATCH_SHA256}"
if [[ ! "$WQ_CHARGE_STOP_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "WQ charge-stop execution/resource identity changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WQ_CHARGE_STOP_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test ! -e "$OUTPUT"

python3 - "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    contract.get("schema") != "wq_llm_charge_stop_paired64_contract_v1"
    or contract.get("identity") != "wq_llm_charge_stop_paired64_v1"
    or contract.get("status") != "authorized_paired64_pilot"
    or contract["resources"]["a800"] != 1
    or contract["resources"]["cpus"] != 8
    or contract["panel"]["attempts_per_arm"] != 64
    or contract["audit_policy"]["mode"] != "minimal_four_artifacts"
):
    raise SystemExit("WQ charge-stop contract changed")
for entry in contract["implementation"].values():
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"WQ charge-stop implementation changed: {path}")
print("wq_charge_stop_installed_source_gate=PASS")
PY

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

python - <<'PY'
import torch

if torch.cuda.device_count() != 1 or "A800" not in torch.cuda.get_device_name(0):
    raise SystemExit("WQ charge-stop exact A800 environment smoke failed")
print(f"wq_charge_stop_device={torch.cuda.get_device_name(0)}")
print(f"wq_charge_stop_torch={torch.__version__}")
print("wq_charge_stop_environment_smoke=PASS")
PY

usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "pipeline_stage=paired64_wq_proposal_only"
python "$RUNNER" \
  --contract "$CONTRACT" \
  --protocol "$PROTOCOL" \
  --gate-a-lock "$GATE" \
  --llama-root "$LLAMA_ROOT" \
  --llama-adapter "$ADAPTER" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$WQ_CHARGE_STOP_PATCH_SHA256"

echo "wq_charge_stop_terminal_acceptance=PASS"
echo "wq_charge_stop_automatic_downstream=false"

#!/bin/bash
#SBATCH --job-name=wq-fplan-1ep-p64-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_formula_plan_sft_one_epoch_v2
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_formula_plan_sft_one_epoch_v2.json
CONTRACT_SHA256=f50b934cb24b026815caea6ac387e4a0604ae32310ee9c46b67e0cc6e1cb0e46
SOURCE="$RUN/outputs/sft_wq_mixed_seed11_ctx512_v3/coarse_sft.jsonl"
SOURCE_SHA256=a4aec58ca904d466b650a8641a3fe527e9e25249e2f8fe7453145e55588bd899
LLAMA_ROOT=/public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct
INITIAL_ADAPTER="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1/epoch_03/adapter_final"
INITIAL_MODEL_SHA256=9e12192fbd9fd42a1cdc236a50a3e70f8f97e4b44353cb37986c72dafd21b45c
INITIAL_CONFIG_SHA256=c8f3fe4407f254e14365e59b2a4a9f7591164ed12c1f9fde2146b67cb56e4c38
DATA="$RUN/outputs/wq_formula_plan_sft_pilot_v1/data"
DATA_SHA256=b96ca8c08ddf1fe65daa9568f1e9507d1afa4d5886794a4e7115c8a08df4e48e
DATA_MANIFEST_SHA256=3eab205be1458fcda087ee3b1cc8b27f262ca352168af3f68499e58185861c92
DATA_TOKEN_AUDIT_SHA256=d0ebe69bdbd3fd7b5bf8ae872a7e60c11736090e4fbf2b497dceb1c5c8d84ac4
BASELINE="$RUN/outputs/wq_llm_charge_stop_paired64_v1/attempts.jsonl"
OUTPUT="$RUN/outputs/$IDENTITY"
TRAIN="$OUTPUT/training"
EVAL="$OUTPUT/paired64"

: "${WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256:?submission must export WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256}"
if [[ ! "$WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "formula-plan one-epoch execution/resource identity changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$DATA/formula_plan_sft.jsonl" | awk '{print $1}')" = "$DATA_SHA256"
test "$(sha256sum "$DATA/manifest.json" | awk '{print $1}')" = "$DATA_MANIFEST_SHA256"
test "$(sha256sum "$DATA/token_audit.json" | awk '{print $1}')" = "$DATA_TOKEN_AUDIT_SHA256"
test ! -e "$OUTPUT"

python3 - "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    contract.get("schema") != "wq_formula_plan_sft_one_epoch_contract_v2"
    or contract.get("identity") != "wq_formula_plan_sft_one_epoch_v2"
    or contract.get("status") != "authorized_one_epoch_reachability_pilot"
    or contract["resources"]["a800"] != 1
    or contract["resources"]["cpus"] != 8
    or contract["resources"]["cpus"] > 8 * contract["resources"]["a800"]
    or contract["optimizer"]["epochs"] != 1.0
    or contract["panel"]["attempts"] != 64
    or contract["audit_policy"]["mode"] != "minimal_terminal_only"
):
    raise SystemExit("formula-plan one-epoch contract changed")
for entry in contract["implementation"].values():
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"formula-plan implementation changed: {path}")
print("wq_formula_plan_one_epoch_installed_source_gate=PASS")
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
    raise SystemExit("formula-plan one-epoch exact A800 smoke failed")
print(f"wq_formula_plan_one_epoch_device={torch.cuda.get_device_name(0)}")
print(f"wq_formula_plan_one_epoch_torch={torch.__version__}")
print("wq_formula_plan_one_epoch_environment_smoke=PASS")
PY

python tests/test_crysllmgen_formula_plan.py -v
python tests/test_crysllmgen_gate_lock.py -v

usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "pipeline_stage=formula_plan_one_epoch_training"
python scripts/a800/train_wq_formula_plan_lora.py \
  --data-dir "$DATA" \
  --source-sha256 "$SOURCE_SHA256" \
  --llama-root "$LLAMA_ROOT" \
  --initial-adapter "$INITIAL_ADAPTER" \
  --initial-adapter-model-sha256 "$INITIAL_MODEL_SHA256" \
  --initial-adapter-config-sha256 "$INITIAL_CONFIG_SHA256" \
  --output-dir "$TRAIN" \
  --execution-patch-sha256 "$WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --identity "$IDENTITY" \
  --report-schema wq_formula_plan_sft_training_report_v2 \
  --data-identity wq_formula_plan_sft_pilot_v1 \
  --expected-data-sha256 "$DATA_SHA256" \
  --expected-manifest-sha256 "$DATA_MANIFEST_SHA256" \
  --expected-token-audit-sha256 "$DATA_TOKEN_AUDIT_SHA256" \
  --max-steps -1 \
  --num-train-epochs 1.0 \
  --minimum-completed-updates 959 \
  --maximum-completed-updates 960 \
  --learning-rate 1.0e-5

echo "pipeline_stage=formula_plan_one_epoch_paired64"
python scripts/a800/run_wq_formula_plan_paired64_v1.py \
  --contract "$CONTRACT" \
  --contract-sha256 "$CONTRACT_SHA256" \
  --baseline-attempts "$BASELINE" \
  --llama-root "$LLAMA_ROOT" \
  --trained-adapter "$TRAIN/adapter_final" \
  --output-dir "$EVAL" \
  --execution-patch-sha256 "$WQ_FORMULA_PLAN_ONE_EPOCH_PATCH_SHA256" \
  --identity "$IDENTITY" \
  --contract-schema wq_formula_plan_sft_one_epoch_contract_v2 \
  --contract-status authorized_one_epoch_reachability_pilot \
  --training-report-schema wq_formula_plan_sft_training_report_v2 \
  --row-schema wq_formula_plan_one_epoch_paired_attempt_v2 \
  --report-schema wq_formula_plan_one_epoch_paired64_terminal_v2 \
  --minimum-training-updates 959 \
  --maximum-training-updates 960 \
  --minimum-training-epochs 0.999

echo "wq_formula_plan_one_epoch_terminal_acceptance=PASS"
echo "wq_formula_plan_one_epoch_automatic_downstream=false"

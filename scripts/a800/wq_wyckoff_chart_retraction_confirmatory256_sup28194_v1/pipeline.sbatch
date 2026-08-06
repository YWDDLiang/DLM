#!/bin/bash
#SBATCH --job-name=wq-wtb256-sup28194-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
EXECUTION_IDENTITY=wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1
SCIENTIFIC_IDENTITY=wq_wyckoff_chart_retraction_confirmatory256_v1
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_chart_retraction_confirmatory256_v1.json
CONTRACT_SHA256=293c026d2f371b592a81e8e4d3982b4cb65ae3b0d90b82bf72a639caae24b77a
AUTHORIZATION=runs/remote_audit/20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/submission_authorization_record.json
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
ADAPTER_TRAINING_PATCH=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491
PROTOCOL=configs/experiments/wyckoff_codiffusion/protocol_v4.yaml
GATE="$RUN/outputs/gate_a_lock_v8.json"
LLAMA_ROOT=/public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct
ADAPTER="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1/epoch_03/adapter_final"
PARENT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
OUTPUT="$RUN/outputs/$EXECUTION_IDENTITY"
SOURCES="$OUTPUT/sources"
ARMS="$OUTPUT/arms"
EVALUATION="$OUTPUT/evaluation"
SUMMARY="$OUTPUT/summary"

: "${WTB256_SUP28194_EXECUTION_PATCH_SHA256:?submission must export WTB256_SUP28194_EXECUTION_PATCH_SHA256}"
: "${WTB256_SUP28194_AUTHORIZATION_SHA256:?submission must export WTB256_SUP28194_AUTHORIZATION_SHA256}"
if [[ ! "$WTB256_SUP28194_EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB256_SUP28194_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "WTB-256 sup28194 execution/resource identity changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB256_SUP28194_EXECUTION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = \
  "$WTB256_SUP28194_AUTHORIZATION_SHA256"
test ! -e "$OUTPUT"

python3 - "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    contract.get("identity")
    != "wq_wyckoff_chart_retraction_confirmatory256_v1"
    or contract.get("status")
    != "local_built_remote_execution_not_authorized"
    or contract["resources"] != {
        "partition": "gpu",
        "nodes": 1,
        "tasks": 1,
        "a800": 1,
        "cpus": 8,
        "memory_gib": 96,
        "time_limit": "18:00:00",
        "cpu_per_a800_gate": "cpus<=8*a800",
    }
):
    raise SystemExit("WTB-256 scientific contract/resource identity changed")
for entry in contract["implementation"].values():
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"WTB-256 installed scientific source changed: {path}")
print("wtb256_sup28194_installed_scientific_source_hash_gate=PASS")
PY

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
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
export CRYSLLMGEN_METRICS_NUM_CPUS=1
export PYTHONHASHSEED=0

python - <<'PY'
import importlib.metadata
import os
import sys
import torch

if (
    os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff"
    or sys.version_info[:2] != (3, 10)
    or torch.__version__ != "2.4.0+cu121"
    or importlib.metadata.version("torch-scatter") != "2.1.2+pt24cu121"
    or torch.cuda.device_count() != 1
    or "A800" not in torch.cuda.get_device_name(0)
):
    raise SystemExit("WTB-256 sup28194 exact environment/A800 smoke failed")
print(f"wtb256_sup28194_device={torch.cuda.get_device_name(0)}")
print("wtb256_sup28194_environment_smoke=PASS")
PY

mkdir "$OUTPUT"
usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "pipeline_stage=freeze_wq_sources_256"
python scripts/a800/run_wq_wyckoff_chart_retraction_sources256_v1.py \
  --contract "$CONTRACT" \
  --protocol "$PROTOCOL" \
  --gate-a-lock "$GATE" \
  --llama-root "$LLAMA_ROOT" \
  --llama-adapter "$ADAPTER" \
  --output-dir "$SOURCES" \
  --execution-patch-sha256 "$WTB256_SUP28194_EXECUTION_PATCH_SHA256"

echo "pipeline_stage=paired_R_U_T_generation"
python scripts/a800/run_wq_wyckoff_chart_retraction_arms256_v1.py \
  --contract "$CONTRACT" \
  --source-jsonl "$SOURCES/source_attempts.jsonl" \
  --source-report "$SOURCES/source_report.json" \
  --checkpoint "$PARENT" \
  --output-dir "$ARMS" \
  --execution-patch-sha256 "$WTB256_SUP28194_EXECUTION_PATCH_SHA256"

echo "pipeline_stage=crysllmgen_direct_metrics"
mkdir "$EVALUATION"
for arm in R U T; do
  mkdir "$EVALUATION/$arm"
  lower="$(printf '%s' "$arm" | tr '[:upper:]' '[:lower:]')"
  python scripts/a800/run_crysllmgen_metrics.py \
    --generation-jsonl "$ARMS/${lower}_generation.jsonl" \
    --gt-csv reference/crysllmgen/data/mp_20/test.csv \
    --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
    --output-dir "$EVALUATION/$arm/crysllmgen_metrics" \
    --eval-model-name mp20
done

echo "pipeline_stage=exact_R5C_A100_protocol_on_A800_SUN"
for arm in R U T; do
  lower="$(printf '%s' "$arm" | tr '[:upper:]' '[:lower:]')"
  python scripts/a800/run_crysllmgen_a100_sun.py \
    --generation-jsonl "$ARMS/${lower}_generation.jsonl" \
    --expected-attempts 256 \
    --eval-sun-py reference/a100_eval_sun/eval_sun.py \
    --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
    --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
    --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
    --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
    --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/crysllm/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
    --base-source-bundle-sha256 "$BASE_SOURCE" \
    --execution-patch-sha256 "$WTB256_SUP28194_EXECUTION_PATCH_SHA256" \
    --output-dir "$EVALUATION/$arm/r5c_a100_sun"
done

echo "pipeline_stage=paired_summary_and_promotion_lock"
python scripts/a800/summarize_wq_wyckoff_chart_retraction_confirmatory256_v1.py \
  --contract "$CONTRACT" \
  --arms-dir "$ARMS" \
  --evaluation-dir "$EVALUATION" \
  --output-dir "$SUMMARY"

python - "$OUTPUT" "$CONTRACT_SHA256" \
  "$WTB256_SUP28194_EXECUTION_PATCH_SHA256" \
  "$WTB256_SUP28194_AUTHORIZATION_SHA256" "$usage_csv" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
contract_sha, patch_sha, authorization_sha = sys.argv[2:5]
gpu_csv = Path(sys.argv[5]).resolve()

def read(relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))

def identity(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

source = read("sources/source_report.json")
arms = read("arms/arms_report.json")
summary = read("summary/summary.json")
lock = read("summary/promotion_lock.json")
if (
    source.get("ok") is not True
    or source.get("terminal") != 256
    or arms.get("ok") is not True
    or any(
        values.get("succeeded", 0) + values.get("failed", 0) != 256
        for values in arms.get("terminal_counts", {}).values()
    )
    or summary.get("integrity_pass") is not True
    or lock.get("summary_sha256")
    != identity(root / "summary/summary.json")["sha256"]
    or lock.get("automatic_training_authorized") is not False
):
    raise SystemExit("WTB-256 sup28194 terminal acceptance failed")
payload = {
    "schema": "wq_wyckoff_chart_retraction_confirmatory256_sup28194_terminal_v1",
    "ok": True,
    "acceptance": "PASS",
    "execution_identity": (
        "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
    ),
    "scientific_identity": "wq_wyckoff_chart_retraction_confirmatory256_v1",
    "supersedes_failed_job_id": 28194,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "contract_sha256": contract_sha,
    "execution_patch_sha256": patch_sha,
    "authorization_record_sha256": authorization_sha,
    "source_report": identity(root / "sources/source_report.json"),
    "arms_report": identity(root / "arms/arms_report.json"),
    "summary": identity(root / "summary/summary.json"),
    "promotion_lock": identity(root / "summary/promotion_lock.json"),
    "gpu_csv": identity(gpu_csv),
    "decision": summary["decision"],
    "scientific_promotion_pass": summary["scientific_promotion_pass"],
    "retry_or_replacement_used": False,
    "training_performed": False,
    "automatic_followup_submission": False,
}
path = root / "terminal_acceptance.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("wtb256_sup28194_terminal_acceptance=PASS")
print(f"wtb256_sup28194_decision={summary['decision']}")
PY

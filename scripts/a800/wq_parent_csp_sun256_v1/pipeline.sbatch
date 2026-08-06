#!/bin/bash
#SBATCH --job-name=wq-parent-sun256-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_sun256_v1.json
PLAN_SHA256=51a6047aa8a5063852c2b2569dfd02953b1b1038b683266012262e9aed7401c6
AUTHORIZATION=runs/remote_audit/20260724_wq_parent_csp_sun256_v1/authorization_record.json
AUTHORIZATION_SHA256=1da13c5b5623c67314cd042e2204c9adaa3d3eafcf0e3919da263ca1e5f962db
P0_AUDIT=runs/remote_audit/20260724_wq_parent_csp_probe_sup27407_job27409_terminal_audit_v1.json
P0_AUDIT_SHA256=04749ac74f617567cdcb43f4f86ae8c91ee5048ec1cca82ec870c2936926c824
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
ADAPTER_TRAINING_PATCH_SHA256=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
ADAPTER="$TRAIN/epoch_03/adapter_final"
TRAINING_REPORT="$TRAIN/epoch_03/training_report.json"
OUTPUT="$RUN/outputs/wq_parent_csp_sun256_v1"
SOURCE_GENERATION="$OUTPUT/source_generation.jsonl"
GENERATION="$OUTPUT/generation.jsonl"

: "${EVALUATION_PATCH_SHA256:?submission must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]; then
  echo "parent-CSP SUN256 identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH_SHA256}.json"
test "$(sha256sum "$PLAN" | awk '{print $1}')" = "$PLAN_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$P0_AUDIT" | awk '{print $1}')" = "$P0_AUDIT_SHA256"
test "$(sha256sum "$TRAINING_REPORT" | awk '{print $1}')" = \
  1d4bb3275dd5f1a20b9743161b59c633faad6f008ebbf7881ad07ddd2c6dfaf9
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')" = \
  9e12192fbd9fd42a1cdc236a50a3e70f8f97e4b44353cb37986c72dafd21b45c
test ! -e "$OUTPUT"

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
    raise SystemExit("parent-CSP SUN256 requires exactly one A800")
print(f"device={torch.cuda.get_device_name(0)}")
print("pipeline_stage=generate_256")
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

python scripts/a800/probe_wq_proposal_with_parent_csp.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --gate-a-lock "$RUN/outputs/gate_a_lock_v8.json" \
  --csp-checkpoint /public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt \
  --llama-root /public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --llama-adapter "$ADAPTER" \
  --output-jsonl "$SOURCE_GENERATION" \
  --attempt-ledger "$OUTPUT/attempt_ledger.jsonl" \
  --report "$OUTPUT/generation_report.json" \
  --experiment-id wq-parent-csp-sun256-v1 \
  --pairing-id wq-epoch-selection-paired-noise-sup27104-v2 \
  --training-seed 11 \
  --sampling-seed 101 \
  --attempts 256 \
  --start-ordinal 256 \
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --diagnostic-execution-patch-sha256 "$EVALUATION_PATCH_SHA256"

python - "$SOURCE_GENERATION" "$OUTPUT/generation_report.json" \
  "$EVALUATION_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

generation_path, report_path = map(Path, sys.argv[1:3])
evaluation_patch = sys.argv[3]
rows = [
    json.loads(line)
    for line in generation_path.read_text(encoding="utf-8").splitlines()
    if line
]
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    len(rows) != 256
    or [row.get("ordinal") for row in rows] != list(range(256, 512))
    or {row.get("method") for row in rows}
    != {"DIAG-WQ-PROPOSAL-PARENT-CSP32"}
    or any(row.get("status") not in {"succeeded", "failed"} for row in rows)
    or any(row.get("retry_or_replacement_used") is not False for row in rows)
    or any(
        row.get("diagnostic_execution_patch_sha256") != evaluation_patch
        for row in rows
    )
    or report.get("schema") != "wq_parent_csp_probe_report_v1"
    or report.get("ok") is not True
    or report.get("submitted") != 256
    or report.get("terminal") != 256
    or report.get("succeeded", 0) + report.get("failed", 0) != 256
    or report.get("reverse_steps") != 32
    or report.get("retry_or_replacement_used") is not False
    or report.get("diagnostic_execution_patch_sha256") != evaluation_patch
):
    raise SystemExit("parent-CSP SUN256 generation acceptance failed")
print("parent_csp_sun256_generation_acceptance=PASS")
print(f"generation_succeeded={report['succeeded']}")
print(f"generation_failed={report['failed']}")
PY

echo "pipeline_stage=adapt_and_crysllmgen_metrics"
conda activate crysllm
export PYTHONPATH="$ROOT"
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CRYSLLMGEN_METRICS_NUM_CPUS=1

python scripts/a800/adapt_wq_parent_probe_for_eval.py \
  --source-jsonl "$SOURCE_GENERATION" \
  --output-jsonl "$GENERATION" \
  --manifest-json "$OUTPUT/evaluation_adapter_manifest.json" \
  --expected-attempts 256 \
  --expected-start-ordinal 256 \
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --evaluation-execution-patch-sha256 "$EVALUATION_PATCH_SHA256"

python scripts/a800/run_crysllmgen_metrics.py \
  --generation-jsonl "$GENERATION" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
  --output-dir "$OUTPUT/crysllmgen_metrics" \
  --eval-model-name mp20

echo "pipeline_stage=strict_and_meta_SUN"
python scripts/a800/run_crysllmgen_a100_sun.py \
  --generation-jsonl "$GENERATION" \
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
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --output-dir "$OUTPUT/r5c_a100_sun"

python - "$OUTPUT" "$PLAN_SHA256" "$EVALUATION_PATCH_SHA256" \
  "$usage_csv" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
plan_sha, execution_patch = sys.argv[2:4]
usage_csv = Path(sys.argv[4]).resolve()

def read_json(relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))

def sha(relative):
    path = root / relative
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

generation = read_json("generation_report.json")
adapter = read_json("evaluation_adapter_manifest.json")
metrics = read_json("crysllmgen_metrics/report.json")
sun = read_json("r5c_a100_sun/attempt_summary.json")
if (
    generation.get("ok") is not True
    or generation.get("terminal") != 256
    or adapter.get("ok") is not True
    or adapter.get("attempts") != 256
    or adapter.get("start_ordinal") != 256
    or adapter.get("end_ordinal_inclusive") != 511
    or adapter.get("retry_or_replacement_used") is not False
    or metrics.get("ok") is not True
    or metrics.get("attempts") != 256
    or metrics.get("denominator") != "all_generation_attempts"
    or metrics.get("retry_or_replacement_used") is not False
    or sun.get("ok") is not True
    or (sun.get("counts") or {}).get("total_attempts") != 256
    or sun.get("denominator") != "all_generation_attempts"
    or sun.get("execution_patch_sha256") != execution_patch
    or sun.get("retry_or_replacement_used") is not False
):
    raise SystemExit("parent-CSP SUN256 terminal acceptance failed")

payload = {
    "schema": "wq_parent_csp_sun256_terminal_acceptance_v1",
    "ok": True,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "plan_sha256": plan_sha,
    "evaluation_execution_patch_sha256": execution_patch,
    "generation": {
        "submitted": generation["submitted"],
        "terminal": generation["terminal"],
        "succeeded": generation["succeeded"],
        "failed": generation["failed"],
        "report_sha256": sha("generation_report.json"),
        "source_generation_sha256": sha("source_generation.jsonl"),
        "attempt_ledger_sha256": sha("attempt_ledger.jsonl"),
    },
    "evaluation_adapter": {
        "manifest_sha256": sha("evaluation_adapter_manifest.json"),
        "generation_sha256": sha("generation.jsonl"),
        "mapping": adapter["structure_mapping"],
    },
    "crysllmgen_direct_metrics": {
        "attempts": metrics["attempts"],
        "generation_succeeded": metrics["generation_succeeded"],
        "comp_valid_count": metrics["comp_valid_count"],
        "struct_valid_count": metrics["struct_valid_count"],
        "valid_count": metrics["valid_count"],
        "metrics_unchanged_upstream": metrics["metrics_unchanged_upstream"],
        "report_sha256": sha("crysllmgen_metrics/report.json"),
        "attempt_metrics_sha256": sha("crysllmgen_metrics/attempt_metrics.jsonl"),
    },
    "sun": {
        "thresholds_ev_per_atom": sun["thresholds_ev_per_atom"],
        "counts": sun["counts"],
        "rates": sun["rates"],
        "coverage_adjusted_selection_role": (
            sun["coverage_adjusted_selection_role"]
        ),
        "summary_sha256": sha("r5c_a100_sun/attempt_summary.json"),
        "attempt_results_sha256": sha("r5c_a100_sun/attempt_results.jsonl"),
    },
    "gpu_csv": {
        "path": str(usage_csv),
        "sha256": hashlib.sha256(usage_csv.read_bytes()).hexdigest(),
    },
    "attempt_retry_or_replacement_used": False,
    "long_training_started": False,
}
path = root / "terminal_acceptance.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("parent_csp_sun256_terminal_acceptance=PASS")
print(json.dumps(payload, sort_keys=True))
PY

#!/bin/bash
#SBATCH --job-name=wq-parent-eval-sup27410-v1
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
PLAN=configs/experiments/wyckoff_codiffusion/wq_parent_csp_sun256_eval_sup27410_v1.json
AUTHORIZATION=runs/remote_audit/20260724_wq_parent_csp_sun256_eval_sup27410_v1/authorization_record.json
FAILURE_AUDIT=runs/remote_audit/20260724_wq_parent_csp_sun256_job27410_terminal_failure_v1.json
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1"
GENERATION="$SOURCE/generation.jsonl"
OUTPUT="$RUN/outputs/wq_parent_csp_sun256_eval_sup27410_v1"
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
GENERATION_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
GENERATION_REPORT_SHA256=02307d57e863b50d88bcd52f147aad9217f71bdf0343b55013c39a47d468d95c
ADAPTER_MANIFEST_SHA256=38c4623e3035fbfe858f1f25952e66881431bc0c28f7589fabfeb1219dc0dd84
FAILURE_AUDIT_SHA256=2937e3a98ab761811023b4d20c18a051379ddd59f10b3ae54be8eb11273df78d
DIFF_ENV=/public/home/jiaosz/miniconda3/envs/diff_meets_diff
CHGNET_RUNTIME="$DIFF_ENV/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"

: "${EVALUATION_PATCH_SHA256:?submission must export EVALUATION_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]]; then
  printf '%s\n' "evaluation identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test "$(sha256sum "$GENERATION" | awk '{print $1}')" = "$GENERATION_SHA256"
test "$(sha256sum "$SOURCE/generation_report.json" | awk '{print $1}')" = \
  "$GENERATION_REPORT_SHA256"
test "$(sha256sum "$SOURCE/evaluation_adapter_manifest.json" | awk '{print $1}')" = \
  "$ADAPTER_MANIFEST_SHA256"
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
export CRYSLLMGEN_METRICS_NUM_CPUS=1
export PYTHONHASHSEED=0

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

python - "$OUTPUT/environment_contract.json" "$GENERATION" \
  "$EVALUATION_PATCH_SHA256" "$CHGNET_RUNTIME" <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

import torch
import torch_scatter

contract_path = Path(sys.argv[1])
generation_path = Path(sys.argv[2])
execution_patch = sys.argv[3]
chgnet_checkpoint = Path(sys.argv[4])
expected_prefix = Path(
    "/public/home/jiaosz/miniconda3/envs/diff_meets_diff"
)
snapshot = Path(
    "crystal_dlm/wqcodiff/crysllmgen/upstream"
).resolve()
sys.path.insert(0, str(snapshot))
try:
    from compute_metrics import Crystal, GenEval, get_gt_crys_ori
finally:
    sys.path.pop(0)
if (
    os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff"
    or Path(os.environ.get("CONDA_PREFIX", "")).resolve() != expected_prefix
    or Path(sys.executable).resolve()
    != (expected_prefix / "bin/python3.10").resolve()
    or torch.__version__ != "2.4.0+cu121"
    or torch.version.cuda != "12.1"
    or md.version("torch-scatter") != "2.1.2+pt24cu121"
    or md.version("torch-geometric") != "1.7.2"
    or md.version("chgnet") != "0.4.2"
    or torch.cuda.device_count() != 1
    or "A800" not in torch.cuda.get_device_name(0)
    or not chgnet_checkpoint.is_file()
):
    raise SystemExit("diff_meets_diff evaluation environment smoke failed")
rows = [
    json.loads(line)
    for line in generation_path.read_text(encoding="utf-8").splitlines()
    if line
]
if (
    len(rows) != 256
    or [row.get("ordinal") for row in rows] != list(range(256, 512))
    or any(row.get("status") != "succeeded" for row in rows)
    or len({row.get("attempt_id") for row in rows}) != 256
):
    raise SystemExit("frozen generation panel contract failed")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema": "wq_parent_csp_diff_meets_diff_environment_contract_v1",
    "ok": True,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "execution_patch_sha256": execution_patch,
    "conda_environment": os.environ["CONDA_DEFAULT_ENV"],
    "conda_prefix": os.environ["CONDA_PREFIX"],
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_scatter": md.version("torch-scatter"),
    "torch_scatter_file": torch_scatter.__file__,
    "torch_geometric": md.version("torch-geometric"),
    "chgnet": md.version("chgnet"),
    "chgnet_runtime_checkpoint": str(chgnet_checkpoint.resolve()),
    "chgnet_runtime_checkpoint_sha256": sha(chgnet_checkpoint),
    "compute_metrics_import": "PASS",
    "compute_metrics_symbols": [
        Crystal.__name__,
        GenEval.__name__,
        get_gt_crys_ori.__name__,
    ],
    "cuda_device": torch.cuda.get_device_name(0),
    "generation_jsonl_sha256": sha(generation_path),
    "attempts": len(rows),
    "regenerated": False,
}
with contract_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("diff_meets_diff_environment_smoke=PASS")
print(json.dumps(payload, sort_keys=True))
PY

printf '%s\n' "pipeline_stage=crysllmgen_direct_metrics"
python scripts/a800/run_crysllmgen_metrics.py \
  --generation-jsonl "$GENERATION" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root crystal_dlm/wqcodiff/crysllmgen/upstream \
  --output-dir "$OUTPUT/crysllmgen_metrics" \
  --eval-model-name mp20

printf '%s\n' "pipeline_stage=strict_and_meta_SUN"
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
  --chgnet-runtime-checkpoint "$CHGNET_RUNTIME" \
  --base-source-bundle-sha256 "$BASE_SOURCE" \
  --execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --output-dir "$OUTPUT/r5c_a100_sun"

python - "$OUTPUT" "$SOURCE" "$PLAN" "$AUTHORIZATION" \
  "$FAILURE_AUDIT" "$EVALUATION_PATCH_SHA256" "$usage_csv" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output, source, plan, authorization, failure_audit, execution_patch, gpu_csv = (
    Path(value).resolve() if index < 5 or index == 6 else value
    for index, value in enumerate(sys.argv[1:])
)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

environment = read(output / "environment_contract.json")
metrics = read(output / "crysllmgen_metrics/report.json")
sun = read(output / "r5c_a100_sun/attempt_summary.json")
generation = read(source / "generation_report.json")
adapter = read(source / "evaluation_adapter_manifest.json")
if (
    environment.get("ok") is not True
    or environment.get("conda_environment") != "diff_meets_diff"
    or environment.get("attempts") != 256
    or environment.get("regenerated") is not False
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
    raise SystemExit("evaluation supersession terminal acceptance failed")
payload = {
    "schema": "wq_parent_csp_sun256_eval_sup27410_terminal_acceptance_v1",
    "ok": True,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "supersedes_failed_job_id": "27410",
    "evaluation_execution_patch_sha256": execution_patch,
    "environment_contract_sha256": sha(output / "environment_contract.json"),
    "plan_sha256": sha(plan),
    "authorization_record_sha256": sha(authorization),
    "failure_audit_sha256": sha(failure_audit),
    "source_generation": {
        "regenerated": False,
        "attempts": 256,
        "succeeded": generation["succeeded"],
        "failed": generation["failed"],
        "generation_jsonl_sha256": sha(source / "generation.jsonl"),
        "generation_report_sha256": sha(source / "generation_report.json"),
        "adapter_manifest_sha256": sha(
            source / "evaluation_adapter_manifest.json"
        ),
        "mapping": adapter["structure_mapping"],
    },
    "crysllmgen_direct_metrics": {
        "attempts": metrics["attempts"],
        "generation_succeeded": metrics["generation_succeeded"],
        "comp_valid_count": metrics["comp_valid_count"],
        "struct_valid_count": metrics["struct_valid_count"],
        "valid_count": metrics["valid_count"],
        "metrics_unchanged_upstream": metrics["metrics_unchanged_upstream"],
        "report_sha256": sha(output / "crysllmgen_metrics/report.json"),
        "attempt_metrics_sha256": sha(
            output / "crysllmgen_metrics/attempt_metrics.jsonl"
        ),
    },
    "sun": {
        "thresholds_ev_per_atom": sun["thresholds_ev_per_atom"],
        "counts": sun["counts"],
        "rates": sun["rates"],
        "coverage_adjusted_selection_role": (
            sun["coverage_adjusted_selection_role"]
        ),
        "summary_sha256": sha(output / "r5c_a100_sun/attempt_summary.json"),
        "attempt_results_sha256": sha(
            output / "r5c_a100_sun/attempt_results.jsonl"
        ),
    },
    "gpu_csv_sha256": sha(gpu_csv),
    "attempt_retry_or_replacement_used": False,
    "training_started": False,
}
path = output / "terminal_acceptance.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("wq_parent_csp_sun256_eval_sup27410_acceptance=PASS")
print(json.dumps(payload, sort_keys=True))
PY

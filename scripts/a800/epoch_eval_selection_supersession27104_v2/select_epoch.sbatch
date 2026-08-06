#!/bin/bash
#SBATCH --job-name=wq-epoch-select-sup27104-v2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BASE_SOURCE=6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908
EXPECTED_ADAPTER_TRAINING_PATCH=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491
EXPECTED_REFINER_TRAINING_PATCH=a868de4ac99d833628eca08304b6ca6066be58f138b5f03d1dd0e508a61beadc
AMENDMENT=configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
PANEL="$RUN/outputs/epoch_checkpoint_panel_sup27104_v2"
EVIDENCE="$RUN/notes/epoch_checkpoint_evidence_sup27104_v2.json"
OUTPUT="$RUN/outputs/epoch_checkpoint_selection_lock_sup27104_v2.json"

: "${EVALUATION_PATCH_SHA256:?submission must export EVALUATION_PATCH_SHA256}"
: "${ADAPTER_TRAINING_PATCH_SHA256:?submission must export ADAPTER_TRAINING_PATCH_SHA256}"
: "${REFINER_TRAINING_PATCH_SHA256:?submission must export REFINER_TRAINING_PATCH_SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$ADAPTER_TRAINING_PATCH_SHA256" != "$EXPECTED_ADAPTER_TRAINING_PATCH" ] \
  || [ "$REFINER_TRAINING_PATCH_SHA256" != "$EXPECTED_REFINER_TRAINING_PATCH" ] \
  || [ "${SLURM_JOB_PARTITION:-}" != short ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]; then
  echo "epoch-selection supersession identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${REFINER_TRAINING_PATCH_SHA256}.json"
test ! -e "$EVIDENCE"
test ! -e "$OUTPUT"
mkdir -p "$RUN/notes" "$RUN/outputs"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONHASHSEED=0

python scripts/a800/build_crysllmgen_epoch_evidence.py \
  --training-root "$TRAIN" \
  --panel-root "$PANEL" \
  --training-amendment "$AMENDMENT" \
  --base-source-bundle-sha256 "$BASE_SOURCE" \
  --adapter-training-execution-patch-sha256 "$ADAPTER_TRAINING_PATCH_SHA256" \
  --refiner-training-execution-patch-sha256 "$REFINER_TRAINING_PATCH_SHA256" \
  --evaluation-execution-patch-sha256 "$EVALUATION_PATCH_SHA256" \
  --supersedes-epoch-evaluation-array-job-id 27104 \
  --output "$EVIDENCE"

python scripts/a800/select_crysllmgen_epoch.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --training-amendment "$AMENDMENT" \
  --evidence-manifest "$EVIDENCE" \
  --output "$OUTPUT"

python - "$OUTPUT" "$ADAPTER_TRAINING_PATCH_SHA256" \
  "$REFINER_TRAINING_PATCH_SHA256" "$EVALUATION_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
adapter_patch, refiner_patch, evaluation_patch = sys.argv[2:]
supersession = report.get("execution_supersession") or {}
if (
    report.get("ok") is not True
    or report.get("selected_epoch") not in {1, 2, 3}
    or report.get("adapter_training_execution_patch_sha256") != adapter_patch
    or report.get("refiner_training_execution_patch_sha256") != refiner_patch
    or report.get("evaluation_execution_patch_sha256") != evaluation_patch
    or report.get("execution_patch_sha256") != evaluation_patch
    or supersession.get("supersedes_epoch_evaluation_array_job_id") != "27104"
    or report.get("retry_or_replacement_used") is not True
    or report.get("attempt_retry_or_replacement_used") is not False
    or len(report.get("candidates") or ()) != 3
    or report.get("coverage_adjusted_sun_used_for_selection") is not False
    or report.get("final_test_or_final_mlip_used_for_selection") is not False
):
    raise SystemExit("epoch checkpoint selection supersession acceptance failed")
print(f"selected_epoch={report['selected_epoch']}")
print("epoch_checkpoint_selection_supersession_acceptance=PASS")
PY

#!/bin/bash
#SBATCH --job-name=wq-epoch-select-sup26955-v2
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
AMENDMENT=configs/experiments/wyckoff_codiffusion/mixed_edit_three_epoch_v1.json
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
PANEL="$RUN/outputs/epoch_checkpoint_panel_sup26955_v2"
EVIDENCE="$RUN/notes/epoch_checkpoint_evidence_sup26955_v2.json"
OUTPUT="$RUN/outputs/epoch_checkpoint_selection_lock_sup26955_v2.json"

: "${EXECUTION_PATCH_SHA256:?submission must export EXECUTION_PATCH_SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "${SLURM_JOB_PARTITION:-}" != short ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]; then
  echo "supersession epoch selection identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
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
  --execution-patch-sha256 "$EXECUTION_PATCH_SHA256" \
  --output "$EVIDENCE"

python scripts/a800/select_crysllmgen_epoch.py \
  --protocol configs/experiments/wyckoff_codiffusion/protocol_v4.yaml \
  --training-amendment "$AMENDMENT" \
  --evidence-manifest "$EVIDENCE" \
  --output "$OUTPUT"

python - "$OUTPUT" "$EXECUTION_PATCH_SHA256" <<'PY'
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("ok") is not True
    or report.get("selected_epoch") not in {1, 2, 3}
    or report.get("execution_patch_sha256") != sys.argv[2]
    or len(report.get("candidates") or ()) != 3
    or report.get("coverage_adjusted_sun_used_for_selection") is not False
    or report.get("final_test_or_final_mlip_used_for_selection") is not False
):
    raise SystemExit("supersession epoch checkpoint selection acceptance failed")
print(f"selected_epoch={report['selected_epoch']}")
print("epoch_checkpoint_selection_acceptance=PASS")
PY

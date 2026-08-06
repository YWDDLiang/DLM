#!/bin/bash
#SBATCH --job-name=v3-pstar-data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_data_preflight_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_data_preflight_v1/logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_pstar_data_preflight_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
H1_SOURCE="${PROJECT_ROOT}/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base"
P0_CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
TRAIN_SHA="d431dfec1de8c3240dbc5648867be1b4b676fd85276e805a177b9944f3a1a157"
VAL_SHA="59327aa789ae5d2bbb66d8a8f0dc882d594bcc14623aa96ce95076ed1b6fc540"

test -d "${SOURCE_ROOT}"
test -d "${H1_SOURCE}"
test -d "${P0_CHECKPOINT}"
test ! -e "${RUN_ROOT}/data/full"
test ! -e "${RUN_ROOT}/data/smoke32"
test ! -e "${RUN_ROOT}/preflight/full"
test ! -e "${RUN_ROOT}/preflight/smoke32"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

cd "${SOURCE_ROOT}"
python scripts/build_h1a2_lookahead_planner_data.py \
  --source-dir "${H1_SOURCE}" \
  --output-dir "${RUN_ROOT}/data/full" \
  --train-sha256 "${TRAIN_SHA}" \
  --val-sha256 "${VAL_SHA}" \
  --train-count 3200 \
  --val-count 256 \
  --seed 17

python scripts/build_h1a2_lookahead_planner_data.py \
  --source-dir "${H1_SOURCE}" \
  --output-dir "${RUN_ROOT}/data/smoke32" \
  --train-sha256 "${TRAIN_SHA}" \
  --val-sha256 "${VAL_SHA}" \
  --train-count 32 \
  --val-count 32 \
  --seed 17

python scripts/preflight_h1a2_lookahead_planner.py \
  --data-dir "${RUN_ROOT}/data/full" \
  --tokenizer-path "${P0_CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/preflight/full" \
  --expected-train-rows 3200 \
  --expected-val-rows 256 \
  --max-length 768

python scripts/preflight_h1a2_lookahead_planner.py \
  --data-dir "${RUN_ROOT}/data/smoke32" \
  --tokenizer-path "${P0_CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/preflight/smoke32" \
  --expected-train-rows 32 \
  --expected-val-rows 32 \
  --max-length 768

python - "${RUN_ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    "schema": "h1a2_pstar_data_preflight_terminal_v1",
    "status": "complete",
    "automatic_downstream": False,
    "scientific_training_authorized": False,
    "full_data_manifest_sha256": sha(root / "data/full/manifest.json"),
    "smoke_data_manifest_sha256": sha(root / "data/smoke32/manifest.json"),
    "full_preflight_report_sha256": sha(
        root / "preflight/full/preflight_report.json"
    ),
    "smoke_preflight_report_sha256": sha(
        root / "preflight/smoke32/preflight_report.json"
    ),
}
(root / "terminal_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

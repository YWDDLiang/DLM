#!/bin/bash
#SBATCH --job-name=v3-dlm-sidecar
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_sidecar_preflight_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_sidecar_preflight_v1/logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_sidecar_preflight_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
SOURCE_DATA="${PROJECT_ROOT}/data/dlm_sft/mp_20_r5_exact_length"
R5C_CHECKPOINT="${PROJECT_ROOT}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dlm_sidecar_preflight_v1"
SIDECAR_ROOT="${RUN_ROOT}/data/h1a2_r5c_plangraph_sidecar_v2"
PREFLIGHT_ROOT="${RUN_ROOT}/preflight"

test -d "${SOURCE_ROOT}"
test -d "${SOURCE_DATA}"
test -d "${R5C_CHECKPOINT}"
test ! -e "${RUN_ROOT}/source_identity.json"
test ! -e "${SIDECAR_ROOT}"
test ! -e "${PREFLIGHT_ROOT}"
test ! -e "${RUN_ROOT}/terminal_report.json"

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

cd "${SOURCE_ROOT}"

python - "${SOURCE_DATA}" "${RUN_ROOT}/source_identity.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()

expected = {
    "train": {"rows": 27136, "bytes": 111422704},
    "val": {"rows": 9047, "bytes": 37082760},
    "test": {"rows": 9046, "bytes": 37074802},
}
expected_small_sha256 = {
    "_SUCCESS": "8afb81283dfd769b50fe01c49ef997b9c99a9c83c76e39de64d5d7b18be16bd3",
    "stats.json": "f3eda4befe498d948bed75315a1a7567164b6e17528469210e152d01570ae5ca",
    "vocab_tokens.txt": "71d3145913eb3496c82f28c654989e37b0031105f52291483788aca3a8bee56b",
}

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for name, expected_sha256 in expected_small_sha256.items():
    actual = sha(root / name)
    if actual != expected_sha256:
        raise SystemExit(f"frozen source metadata changed: {name}")

success = json.loads((root / "_SUCCESS").read_text(encoding="utf-8"))
if success.get("complete") is not True:
    raise SystemExit("frozen R5-C source is not complete")

split_reports = {}
for split, identity in expected.items():
    path = root / f"{split}.jsonl"
    if path.stat().st_size != identity["bytes"]:
        raise SystemExit(f"{split} byte size changed")
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != identity["rows"]:
        raise SystemExit(f"{split} row denominator changed")
    success_split = success.get("splits", {}).get(split, {})
    if (
        success_split.get("rows_seen") != rows
        or success_split.get("rows_written") != rows
        or success_split.get("failures") != 0
        or success_split.get("formula_match_plan_rate") != 1.0
    ):
        raise SystemExit(f"{split} _SUCCESS identity changed")
    split_reports[split] = {
        "path": str(path),
        "rows": rows,
        "bytes": path.stat().st_size,
    }

payload = {
    "schema": "h1a2_frozen_r5c_source_identity_v1",
    "status": "complete",
    "source_root": str(root),
    "total_rows": sum(item["rows"] for item in split_reports.values()),
    "splits": split_reports,
    "small_file_sha256": expected_small_sha256,
    "automatic_downstream": False,
    "scientific_training_authorized": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python scripts/build_h1a2_sidecar_plangraph_data.py \
  --source-dir "${SOURCE_DATA}" \
  --output-dir "${SIDECAR_ROOT}" \
  --splits train val test

mkdir -p "${PREFLIGHT_ROOT}"
python scripts/preflight_planned_corruption_data.py \
  --data-dir "${SIDECAR_ROOT}" \
  --tokenizer-path "${R5C_CHECKPOINT}" \
  --policy d1 \
  --max-length 382 \
  --splits train,val \
  --corruption-seed 20260731 \
  --mask-smoke-rows 32 \
  --mask-smoke-batch-size 8 \
  --output "${PREFLIGHT_ROOT}/d1.json"

python scripts/preflight_planned_corruption_data.py \
  --data-dir "${SIDECAR_ROOT}" \
  --tokenizer-path "${R5C_CHECKPOINT}" \
  --policy d2 \
  --max-length 382 \
  --splits train,val \
  --corruption-seed 20260731 \
  --mask-smoke-rows 32 \
  --mask-smoke-batch-size 8 \
  --output "${PREFLIGHT_ROOT}/d2.json"

python "${EXECUTION_DIR}/assemble_terminal.py" \
  --dataset-root "${SIDECAR_ROOT}" \
  --source-identity "${RUN_ROOT}/source_identity.json" \
  --d1-report "${PREFLIGHT_ROOT}/d1.json" \
  --d2-report "${PREFLIGHT_ROOT}/d2.json" \
  --output "${RUN_ROOT}/terminal_report.json"

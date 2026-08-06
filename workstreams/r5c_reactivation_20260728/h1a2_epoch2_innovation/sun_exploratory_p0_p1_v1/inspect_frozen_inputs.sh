#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260729_h1a2c_jointchem_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

for arm in P0 P1; do
  arm_root="${RUN_ROOT}/arms/${arm}"
  printf 'ARM %s\n' "${arm}"
  find "${arm_root}" -maxdepth 3 -type f \
    \( -name 'raw_generations.jsonl' -o -name 'sample_metrics.json' \
       -o -name 'checkpoint_selection.json' -o -name 'plan_report.json' \) \
    -printf '%p %s\n' | sort
  "${PYTHON}" - "${arm_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "plan512" / "raw_generations.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
print("RAW_ROWS", len(rows))
print("RAW_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
print("SAMPLE_IDXS_FIRST_LAST", rows[0].get("sample_idx"), rows[-1].get("sample_idx"))
print("ROW_KEYS", sorted(rows[0]))
print("FIRST_ROW", json.dumps(rows[0], sort_keys=True)[:1200])
selection = root / "checkpoint_selection.json"
if selection.exists():
    payload = json.loads(selection.read_text(encoding="utf-8"))
    print("SELECTION", json.dumps(payload.get("selected"), sort_keys=True))
PY
done

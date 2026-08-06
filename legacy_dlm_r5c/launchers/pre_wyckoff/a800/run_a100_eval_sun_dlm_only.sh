#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DLM_PT="${DLM_PT:?DLM_PT is required}"
TRAIN_CSV="${TRAIN_CSV:-reference/crysllmgen/data/mp_20/train.csv}"
EVAL_DIR="${EVAL_DIR:-reference/a100_eval_sun}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  {
    echo "===== COMMAND $(date '+%F %T %Z') ====="
    printf '%q ' "$@"
    echo
  } | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== STATUS ${status} $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  return "${status}"
}

export RUN_ID DLM_PT TRAIN_CSV EVAL_DIR MP_CACHE_PATH GLOBAL_RELAX_CACHE_PATH
export PYTHON_BIN NOTES_DIR OUT_DIR LOG_DIR

test -f "${DLM_PT}"
test -f "${TRAIN_CSV}"
test -f "${EVAL_DIR}/eval_sun.py"
test -f "${EVAL_DIR}/eval_sun_resumable.py"
test -f "${MP_CACHE_PATH}"
test -f "${GLOBAL_RELAX_CACHE_PATH}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "stage": "a100_eval_sun_dlm_only",
    "dlm_pt": os.environ["DLM_PT"],
    "train_csv": os.environ["TRAIN_CSV"],
    "eval_dir": os.environ["EVAL_DIR"],
    "mp_cache_path": os.environ["MP_CACHE_PATH"],
    "global_relax_cache_path": os.environ["GLOBAL_RELAX_CACHE_PATH"],
    "python": os.environ["PYTHON_BIN"],
    "cwd": os.getcwd(),
    "host": platform.node(),
    "user": os.environ.get("USER"),
    "thresholds": {"strict": 0.0, "meta_like": 0.1},
    "baseline_reference_not_rerun": {
        "strict_lower_bound_pct": 9.00,
        "strict_adjusted_pct": 9.31,
        "meta_like_lower_bound_pct": 46.10,
        "meta_like_adjusted_pct": 47.67,
        "novel_unique_pct": 88.10,
        "source_report": "reports/20260531_a100_eval_sun_scripts/a100_eval_sun_rerun_report.md",
    },
}
Path(os.environ["NOTES_DIR"], "a100_eval_sun_dlm_only_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

run_logged "${LOG_DIR}/a100_eval_sun_py_compile.log" \
  "${PYTHON_BIN}" -m py_compile \
    "${EVAL_DIR}/eval_sun.py" \
    "${EVAL_DIR}/eval_sun_resumable.py" \
    scripts/a800/check_a100_eval_sun_cache_missing.py

run_logged "${LOG_DIR}/a100_eval_sun_cache_missing.log" \
  "${PYTHON_BIN}" scripts/a800/check_a100_eval_sun_cache_missing.py \
    --eval-dir "${EVAL_DIR}" \
    --train-csv "${TRAIN_CSV}" \
    --cache-path "${MP_CACHE_PATH}" \
    --run "dlm=${DLM_PT}" \
    --summary-json "${NOTES_DIR}/a100_eval_sun_cache_missing.json"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["NOTES_DIR"], "a100_eval_sun_cache_missing.json").read_text())
run = payload["runs"]["dlm"]
missing_chemsys = int(run.get("missing_chemsys") or 0)
missing_structures = int(run.get("missing_structures") or 0)
if os.environ.get("ALLOW_MISSING_CACHE") != "1" and (missing_chemsys or missing_structures):
    raise SystemExit(
        "A100 MP cache still has missing entries for DLM "
        f"(missing_chemsys={missing_chemsys}, missing_structures={missing_structures}); "
        "enrich cache on login node before running S.U.N."
    )
PY

label_out="${OUT_DIR}/dlm_a100_eval_sun"
mkdir -p "${label_out}"

run_logged "${LOG_DIR}/dlm_a100_eval_sun_lite.log" \
  "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun.py" \
    --gen_file "${DLM_PT}" \
    --train_csv "${TRAIN_CSV}" \
    --skip_stability

run_logged "${LOG_DIR}/dlm_a100_eval_sun_strict.log" \
  "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun_resumable.py" \
    --gen_file "${DLM_PT}" \
    --train_csv "${TRAIN_CSV}" \
    --output_dir "${label_out}" \
    --stable_threshold 0.0 \
    --mp_cache_path "${MP_CACHE_PATH}" \
    --global_relax_cache_path "${GLOBAL_RELAX_CACHE_PATH}"
cp "${label_out}/RESULTS_SUMMARY.md" "${NOTES_DIR}/dlm_a100_eval_sun_strict_summary.md"

run_logged "${LOG_DIR}/dlm_a100_eval_sun_meta_like.log" \
  "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun_resumable.py" \
    --gen_file "${DLM_PT}" \
    --train_csv "${TRAIN_CSV}" \
    --output_dir "${label_out}" \
    --stable_threshold 0.1 \
    --mp_cache_path "${MP_CACHE_PATH}" \
    --global_relax_cache_path "${GLOBAL_RELAX_CACHE_PATH}"
cp "${label_out}/RESULTS_SUMMARY.md" "${NOTES_DIR}/dlm_a100_eval_sun_meta_like_summary.md"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import re
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])

def parse_summary(path):
    payload = {"path": str(path)}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_").replace(".", "")
        value = value.strip()
        payload[key] = value
        m = re.search(r"([-+]?\d+(?:\.\d+)?)%", value)
        if m:
            payload[f"{key}_pct"] = float(m.group(1))
        m = re.search(r"(\d+)\s*/\s*(\d+)", value)
        if m:
            payload[f"{key}_num"] = int(m.group(1))
            payload[f"{key}_den"] = int(m.group(2))
    return payload

summary = {
    "baseline_reference_not_rerun": {
        "strict_lower_bound_pct": 9.00,
        "strict_adjusted_pct": 9.31,
        "meta_like_lower_bound_pct": 46.10,
        "meta_like_adjusted_pct": 47.67,
        "novel_unique_pct": 88.10,
    },
    "dlm_strict": parse_summary(notes / "dlm_a100_eval_sun_strict_summary.md"),
    "dlm_meta_like": parse_summary(notes / "dlm_a100_eval_sun_meta_like_summary.md"),
}
out = notes / "a100_eval_sun_dlm_only_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

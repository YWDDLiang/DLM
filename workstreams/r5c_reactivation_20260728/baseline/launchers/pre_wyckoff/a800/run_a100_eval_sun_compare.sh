#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
BASELINE_PT="${BASELINE_PT:-/public/home/jiaosz/hengzhang/crysllmgen-main/experiments/diff_guided_llm/baseline_5epoch_uncond_1000/M1_mp_20_1000.pt}"
DLM_PT="${DLM_PT:-runs/20260531_0040-r5c-full1000-sun/outputs/r5c_refined1000/dlm_refined_mp_1000.pt}"
TRAIN_CSV="${TRAIN_CSV:-reference/crysllmgen/data/mp_20/train.csv}"
EVAL_DIR="${EVAL_DIR:-reference/a100_eval_sun}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
MP_API_KEY_ENV_NAME="${MP_API_KEY_ENV_NAME:-MP_API_KEY}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

write_config() {
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "stage": "a100_eval_sun_baseline_vs_r5c_dlm",
    "baseline_pt": os.environ["BASELINE_PT"],
    "dlm_pt": os.environ["DLM_PT"],
    "train_csv": os.environ["TRAIN_CSV"],
    "eval_dir": os.environ["EVAL_DIR"],
    "mp_cache_path": os.environ["MP_CACHE_PATH"],
    "global_relax_cache_path": os.environ["GLOBAL_RELAX_CACHE_PATH"],
    "mp_api_key_env": os.environ.get("MP_API_KEY_ENV_NAME"),
    "mp_api_key_file_present": bool(os.environ.get("MP_API_KEY_FILE")),
    "python": os.environ["PYTHON_BIN"],
    "cwd": os.getcwd(),
    "host": platform.node(),
    "user": os.environ.get("USER"),
    "thresholds": {"strict": 0.0, "meta_like": 0.1},
    "notes": (
        "A100 eval_sun scripts are executed unchanged after transfer to A800. "
        "eval_sun.py --skip_stability is SUN-lite; eval_sun_resumable.py is "
        "CHGNet + MP-cache full S.U.N. in the A100 script definition."
    ),
}
Path(os.environ["NOTES_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NOTES_DIR"], "a100_eval_sun_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

export RUN_ID BASELINE_PT DLM_PT TRAIN_CSV EVAL_DIR MP_CACHE_PATH GLOBAL_RELAX_CACHE_PATH
export PYTHON_BIN NOTES_DIR MP_API_KEY_FILE MP_API_KEY_ENV_NAME

test -f "${BASELINE_PT}"
test -f "${DLM_PT}"
test -f "${TRAIN_CSV}"
test -f "${EVAL_DIR}/eval_sun.py"
test -f "${EVAL_DIR}/eval_sun_resumable.py"
test -f "${MP_CACHE_PATH}"
test -f "${GLOBAL_RELAX_CACHE_PATH}"

mp_api_args=()
if [ -n "${MP_API_KEY_FILE}" ]; then
  test -f "${MP_API_KEY_FILE}"
  export "${MP_API_KEY_ENV_NAME}=$(tr -d '\r\n' < "${MP_API_KEY_FILE}")"
  mp_api_args=(--mp_api_key_env "${MP_API_KEY_ENV_NAME}")
fi

write_config

run_logged "${LOG_DIR}/a100_eval_sun_py_compile.log" \
  "${PYTHON_BIN}" -m py_compile \
    "${EVAL_DIR}/eval_sun.py" \
    "${EVAL_DIR}/eval_sun_resumable.py"

for label in baseline dlm; do
  if [ "${label}" = "baseline" ]; then
    pt_path="${BASELINE_PT}"
  else
    pt_path="${DLM_PT}"
  fi

  label_out="${OUT_DIR}/${label}_a100_eval_sun"
  mkdir -p "${label_out}"

  run_logged "${LOG_DIR}/${label}_a100_eval_sun_lite.log" \
    "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun.py" \
      --gen_file "${pt_path}" \
      --train_csv "${TRAIN_CSV}" \
      --skip_stability

  run_logged "${LOG_DIR}/${label}_a100_eval_sun_strict.log" \
    "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun_resumable.py" \
      --gen_file "${pt_path}" \
      --train_csv "${TRAIN_CSV}" \
      --output_dir "${label_out}" \
      --stable_threshold 0.0 \
      ${mp_api_args[@]+"${mp_api_args[@]}"} \
      --mp_cache_path "${MP_CACHE_PATH}" \
      --global_relax_cache_path "${GLOBAL_RELAX_CACHE_PATH}"
  cp "${label_out}/RESULTS_SUMMARY.md" "${NOTES_DIR}/${label}_a100_eval_sun_strict_summary.md"

  run_logged "${LOG_DIR}/${label}_a100_eval_sun_meta_like.log" \
    "${PYTHON_BIN}" "${EVAL_DIR}/eval_sun_resumable.py" \
      --gen_file "${pt_path}" \
      --train_csv "${TRAIN_CSV}" \
      --output_dir "${label_out}" \
      --stable_threshold 0.1 \
      ${mp_api_args[@]+"${mp_api_args[@]}"} \
      --mp_cache_path "${MP_CACHE_PATH}" \
      --global_relax_cache_path "${GLOBAL_RELAX_CACHE_PATH}"
  cp "${label_out}/RESULTS_SUMMARY.md" "${NOTES_DIR}/${label}_a100_eval_sun_meta_like_summary.md"
done

"${PYTHON_BIN}" - <<'PY'
import json
import re
from pathlib import Path
import os

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

summary = {}
for label in ("baseline", "dlm"):
    for mode in ("strict", "meta_like"):
        path = notes / f"{label}_a100_eval_sun_{mode}_summary.md"
        summary[f"{label}_{mode}"] = parse_summary(path)

out = notes / "a100_eval_sun_comparison_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

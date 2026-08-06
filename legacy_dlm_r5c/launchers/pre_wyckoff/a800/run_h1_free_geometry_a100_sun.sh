#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260603_h1_geo_free_executor}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CACHE_ONLY="${CACHE_ONLY:-0}"
LABELS="${LABELS:-ablation_default ablation_no_lattice_volume_mask ablation_no_duplicate_coordinate_mask ablation_default_schedule ablation_no_freeze_plan_composition weighted_lattice_up weighted_coord_up weighted_balanced}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
SOURCE_RUN_DIR="runs/${SOURCE_RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}"

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "===== COMMAND $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  printf '%q ' "$@" | tee -a "${log_file}"
  echo | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== STATUS ${status} $(date '+%F %T %Z') =====" | tee -a "${log_file}"
  return "${status}"
}

activate_conda_env() {
  local env_name="$1"
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${env_name}"
  set -u
}

cleanup_key() {
  case "${MP_API_KEY_FILE:-}" in
    /tmp/freegeo_mp_key_*|/public/home/jiaosz/.cache/codex/freegeo_mp_key_*)
      [ -f "${MP_API_KEY_FILE}" ] && rm -f "${MP_API_KEY_FILE}"
      ;;
  esac
}

trap 'status=$?; cleanup_key; echo "${status}" > "${NOTES_DIR}/exit_status.txt"; date "+%F %T %Z" > "${NOTES_DIR}/end_time.txt"; nvidia-smi > "${NOTES_DIR}/gpu_status_end.txt" 2>&1 || true; exit "${status}"' EXIT

date "+%F %T %Z" > "${NOTES_DIR}/start_time.txt"
{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-none}"
  echo "source_run_id=${SOURCE_RUN_ID}"
  echo "labels=${LABELS}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort | grep -v -i 'api\|key\|token\|secret' > "${NOTES_DIR}/environment_redacted.txt"

python - <<PY
import json
from pathlib import Path

payload = {
    "run_id": "${RUN_ID}",
    "source_run_id": "${SOURCE_RUN_ID}",
    "stage": "h1_free_geometry_a100_sun_refined256",
    "labels": "${LABELS}".split(),
    "a100_eval_dir": "reference/a100_eval_sun",
    "mp_cache_path": "${MP_CACHE_PATH}",
    "global_relax_cache_path": "${GLOBAL_RELAX_CACHE_PATH}",
    "allow_missing_cache": "${ALLOW_MISSING_CACHE}",
    "cache_only": "${CACHE_ONLY}",
    "mp_api_key_file_present": bool("${MP_API_KEY_FILE}"),
    "baseline_reference_not_rerun": {
        "strict_adjusted_pct": 9.31,
        "meta_like_adjusted_pct": 47.67,
        "source_report": "reports/20260531_a100_eval_sun_scripts/a100_eval_sun_rerun_report.md",
    },
    "evaluation_note": (
        "Run A100 eval_sun/eval_sun_resumable on existing refined256 outputs. "
        "No resampling, no extra refinement, no verifier selection."
    ),
}
Path("${NOTES_DIR}/h1_free_geometry_a100_sun_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

test -d "${SOURCE_RUN_DIR}"
test -f "${MP_CACHE_PATH}"
test -f "${GLOBAL_RELAX_CACHE_PATH}"

activate_conda_env "${A100_ENV_NAME}"
python -V | tee -a "${LOG_DIR}/python_version.log"

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    reference/a100_eval_sun/eval_sun.py \
    reference/a100_eval_sun/eval_sun_resumable.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py

for label in ${LABELS}; do
  mapfile -t refined_candidates < <(
    find "${SOURCE_RUN_DIR}/outputs/${label}_refined256" \
      -maxdepth 1 \
      -type f \
      -name 'dlm_refined_mp_*.pt' \
      ! -name '*.rank*.pt' \
      | sort
  )
  if [ "${#refined_candidates[@]}" -eq 0 ]; then
    mapfile -t refined_candidates < <(
      find "${SOURCE_RUN_DIR}/outputs/${label}_refined256" \
        -maxdepth 1 \
        -type f \
        -name 'dlm_refined_mp_*.pt' \
        | sort
    )
  fi
  if [ "${#refined_candidates[@]}" -eq 0 ] || [ ! -f "${refined_candidates[0]}" ]; then
    echo "Missing refined pt for ${label}: ${SOURCE_RUN_DIR}/outputs/${label}_refined256/dlm_refined_mp_*.pt" | tee -a "${LOG_DIR}/missing_refined_pt.log"
    continue
  fi
  refined_pt="${refined_candidates[0]}"
  child_run_id="${RUN_ID}-${label}"
  child_notes="runs/${child_run_id}/notes"
  mkdir -p "${child_notes}"

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${NOTES_DIR}/${label}_a100_sun_summary.json" ]; then
    echo "Skipping completed A100 S.U.N. for ${label}" | tee -a "${LOG_DIR}/${label}_skip.log"
    continue
  fi

  run_logged "${LOG_DIR}/${label}_cache_missing_pre.log" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
      --eval-dir reference/a100_eval_sun \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --run "dlm=${refined_pt}" \
      --summary-json "${NOTES_DIR}/${label}_cache_missing_pre.json"

  missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/${label}_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
  if [ "${missing_count}" != "0" ]; then
    if [ "${ALLOW_MISSING_CACHE}" = "1" ]; then
      echo "A100 MP cache has ${missing_count} missing entries for ${label}; continuing because ALLOW_MISSING_CACHE=1." | tee -a "${LOG_DIR}/${label}_cache_missing_allowed.log"
    elif [ -z "${MP_API_KEY_FILE}" ] || [ ! -s "${MP_API_KEY_FILE}" ]; then
      echo "A100 MP cache missing entries for ${label}, but MP_API_KEY_FILE is empty or missing." >&2
      exit 2
    else
      run_logged "${LOG_DIR}/${label}_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${refined_pt}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/${label}_cache_enrich.json"
    fi
  fi

  if [ "${CACHE_ONLY}" = "1" ]; then
    echo "CACHE_ONLY=1: completed cache preparation for ${label}; skipping A100 S.U.N. execution." | tee -a "${LOG_DIR}/${label}_cache_only.log"
    continue
  fi

  run_logged "${LOG_DIR}/${label}_a100_sun.log" \
    env RUN_ID="${child_run_id}" DLM_PT="${refined_pt}" PYTHON_BIN=python \
      MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
      ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
      bash scripts/a800/run_a100_eval_sun_dlm_only.sh

  cp "${child_notes}/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/${label}_a100_sun_summary.json"
  cp "${child_notes}/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/${label}_a100_sun_strict_summary.md"
  cp "${child_notes}/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/${label}_a100_sun_meta_like_summary.md"
done

export RUN_ID SOURCE_RUN_ID NOTES_DIR SOURCE_RUN_DIR LABELS
python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
source_run_id = os.environ["SOURCE_RUN_ID"]
notes = Path(os.environ["NOTES_DIR"])
source_run_dir = Path(os.environ["SOURCE_RUN_DIR"])
source_notes = source_run_dir / "notes"
source_outputs = source_run_dir / "outputs"
labels = os.environ["LABELS"].split()

def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def parse_summary(summary):
    if not summary:
        return {}
    strict = summary.get("dlm_strict", {})
    meta = summary.get("dlm_meta_like", {})
    return {
        "strict_adjusted_pct": strict.get("coverage-adjusted_sun_estimate_pct") or strict.get("adjusted_estimate_pct"),
        "strict_lower_bound_pct": strict.get("full_sun_lower-bound_pct") or strict.get("lower_bound_pct"),
        "strict_stable_evaluated_pct": strict.get("stable_pct") or strict.get("stable_among_evaluated_pct"),
        "meta_like_adjusted_pct": meta.get("coverage-adjusted_sun_estimate_pct") or meta.get("adjusted_estimate_pct"),
        "meta_like_lower_bound_pct": meta.get("full_sun_lower-bound_pct") or meta.get("lower_bound_pct"),
        "meta_like_stable_evaluated_pct": meta.get("stable_pct") or meta.get("stable_among_evaluated_pct"),
        "novel_unique_pct": (
            strict.get("novel_+_unique_pct")
            or strict.get("novel_unique_pct")
            or strict.get("novel_+_unique")
            or strict.get("novel_unique")
        ),
        "strict_e_hull_evaluated": strict.get("e_hull_evaluated"),
        "meta_e_hull_evaluated": meta.get("e_hull_evaluated"),
    }

def preferred_refined_pt(label):
    refined_dir = source_outputs / f"{label}_refined256"
    candidates = sorted(refined_dir.glob("dlm_refined_mp_*.pt"))
    combined = [path for path in candidates if ".rank" not in path.name]
    if combined:
        return str(combined[0])
    if candidates:
        return str(candidates[0])
    return ""

items = []
for label in labels:
    summary = read_json(notes / f"{label}_a100_sun_summary.json")
    sample = read_json(source_outputs / f"{label}_sample256" / "sample_metrics.json")
    crys_payload = read_json(source_notes / f"{label}_crysllmgen_metrics.json")
    crys = crys_payload.get("metrics", crys_payload) if isinstance(crys_payload, dict) else None
    item = {
        "label": label,
        "source_refined_pt": preferred_refined_pt(label),
        "sample_metrics": sample,
        "crysllmgen_metrics": crys,
        "a100_summary": summary,
        "a100_flat": parse_summary(summary),
    }
    items.append(item)

aggregate = {
    "run_id": run_id,
    "source_run_id": source_run_id,
    "items": items,
    "baseline_reference_not_rerun": {
        "strict_adjusted_pct": 9.31,
        "meta_like_adjusted_pct": 47.67,
    },
}
(notes / "h1_free_geometry_a100_sun_aggregate.json").write_text(
    json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# H1 Free-Geometry A100 S.U.N. Aggregate",
    "",
    f"- RUN_ID: `{run_id}`",
    f"- source_RUN_ID: `{source_run_id}`",
    "- Scope: existing refined256 outputs only; no resampling, no extra refinement, no verifier selection.",
    "- Baseline reference is not rerun: strict adjusted `9.31%`, meta-like adjusted `47.67%`.",
    "",
    "| label | n refined | Crys cov_recall | strict adjusted | meta-like adjusted | Novel+Unique | hull eval |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
]
for item in items:
    flat = item["a100_flat"]
    sample = item.get("sample_metrics") or {}
    crys = item.get("crysllmgen_metrics") or {}
    n_refined = sample.get("valid_array_count") or sample.get("graph_success")
    lines.append(
        "| {label} | {n} | {cov} | {strict} | {meta} | {nu} | {hull} |".format(
            label=item["label"],
            n=n_refined,
            cov=crys.get("cov_recall"),
            strict=flat.get("strict_adjusted_pct"),
            meta=flat.get("meta_like_adjusted_pct"),
            nu=flat.get("novel_unique_pct"),
            hull=flat.get("strict_e_hull_evaluated"),
        )
    )
(notes / "h1_free_geometry_a100_sun_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(aggregate, indent=2, sort_keys=True))
PY

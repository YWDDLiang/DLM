#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DLM_MODEL_PATH="${DLM_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
R5C_DLM_CHECKPOINT="${R5C_DLM_CHECKPOINT:-runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final}"
WEIGHTED_SOURCE_RUN="${WEIGHTED_SOURCE_RUN:-runs/20260603_h1_geo_free_executor}"
H1A2_PLANS_JSONL="${H1A2_PLANS_JSONL:-runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200/plans_for_dlm.jsonl}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
LABELS="${LABELS:-ablation_default ablation_no_lattice_volume_mask ablation_no_duplicate_coordinate_mask ablation_default_schedule weighted_lattice_up weighted_coord_up weighted_balanced}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1200}"
REFINE_MAX_PROPOSALS="${REFINE_MAX_PROPOSALS:-1000}"
GPU_COUNT="${GPU_COUNT:-2}"
DLM_NPROC="${DLM_NPROC:-${GPU_COUNT}}"
REFINE_NPROC="${REFINE_NPROC:-${GPU_COUNT}}"
DLM_BATCH_SIZE="${DLM_BATCH_SIZE:-8}"
DLM_TEMPERATURE="${DLM_TEMPERATURE:-0.7}"
DIFF_STEPS="${DIFF_STEPS:-800}"
MAIN_ENV_NAME="${MAIN_ENV_NAME:-diff_meets_diff}"
A100_ENV_NAME="${A100_ENV_NAME:-crysllm}"
CONDA_SH="${CONDA_SH:-/public/home/jiaosz/miniconda3/etc/profile.d/conda.sh}"
MP_CACHE_PATH="${MP_CACHE_PATH:-data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl}"
GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH:-data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl}"
MP_API_KEY_FILE="${MP_API_KEY_FILE:-}"
ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [ "${GPU_COUNT}" -gt 2 ] || [ "${DLM_NPROC}" -gt 2 ] || [ "${REFINE_NPROC}" -gt 2 ]; then
  echo "GPU counts must be <=2 for this project." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUT_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((32000 + (${SLURM_JOB_ID:-0} % 8000)))}"
PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((BASE_MASTER_PORT + PORT_OFFSET))
  PORT_OFFSET=$((PORT_OFFSET + 1))
}

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
    /tmp/freegeo_full1000_batch_mp_key_*|/public/home/jiaosz/.cache/codex/freegeo_full1000_batch_mp_key_*)
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
  echo "labels=${LABELS}"
} > "${NOTES_DIR}/host_user_pwd.txt"
nvidia-smi > "${NOTES_DIR}/gpu_status_start.txt" 2>&1 || true
env | sort | grep -v -i 'api\|key\|token\|secret' > "${NOTES_DIR}/environment_redacted.txt"

python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "stage": "h1_free_geometry_full1000_remaining_a100_sun",
    "labels": "${LABELS}".split(),
    "sample_count": int("${SAMPLE_COUNT}"),
    "refine_max_proposals": int("${REFINE_MAX_PROPOSALS}"),
    "r5c_dlm_checkpoint": "${R5C_DLM_CHECKPOINT}",
    "weighted_source_run": "${WEIGHTED_SOURCE_RUN}",
    "h1a2_plans_jsonl": "${H1A2_PLANS_JSONL}",
    "baseline_reference_not_rerun": {
        "strict_adjusted_pct": 9.31,
        "meta_like_adjusted_pct": 47.67,
    },
    "note": "Runs the seven full1000 branches not covered by the separate no-freeze full1000 job.",
}
Path("${NOTES_DIR}/h1_free_geometry_full1000_remaining_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

activate_conda_env "${MAIN_ENV_NAME}"
python -V | tee -a "${LOG_DIR}/python_version_main.log"

test -f "${H1A2_PLANS_JSONL}"
test -d "${R5C_DLM_CHECKPOINT}"
test -f "${CRYSLLMGEN_CHECKPOINT}"

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/sample_llada_r5_exact_length.py \
    scripts/refine_dlm_with_crysllmgen.py \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/evaluate_r5b_gate.py \
    scripts/a800/check_a100_eval_sun_cache_missing.py \
    scripts/a800/enrich_a100_eval_sun_mp_cache.py \
    reference/a100_eval_sun/eval_sun.py \
    reference/a100_eval_sun/eval_sun_resumable.py

echo "plan_lines=$(wc -l < "${H1A2_PLANS_JSONL}")" | tee "${NOTES_DIR}/plan_lines.txt"

checkpoint_for_label() {
  case "$1" in
    weighted_lattice_up) echo "${WEIGHTED_SOURCE_RUN}/outputs/weighted_lattice_up_sft/final" ;;
    weighted_coord_up) echo "${WEIGHTED_SOURCE_RUN}/outputs/weighted_coord_up_sft/final" ;;
    weighted_balanced) echo "${WEIGHTED_SOURCE_RUN}/outputs/weighted_balanced_sft/final" ;;
    *) echo "${R5C_DLM_CHECKPOINT}" ;;
  esac
}

sample_args_for_label() {
  case "$1" in
    ablation_default)
      SAMPLE_ARGS=(--freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan)
      ;;
    ablation_no_lattice_volume_mask)
      SAMPLE_ARGS=(--freeze-plan-composition --duplicate-coordinate-mask --no-lattice-volume-mask --generation-schedule exact-plan)
      ;;
    ablation_no_duplicate_coordinate_mask)
      SAMPLE_ARGS=(--freeze-plan-composition --no-duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan)
      ;;
    ablation_default_schedule)
      SAMPLE_ARGS=(--freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule default)
      ;;
    weighted_lattice_up|weighted_coord_up|weighted_balanced)
      SAMPLE_ARGS=(--freeze-plan-composition --duplicate-coordinate-mask --lattice-volume-mask --generation-schedule exact-plan)
      ;;
    *)
      echo "Unknown label $1" >&2
      exit 2
      ;;
  esac
}

run_one_label() {
  local label="$1"
  local checkpoint
  checkpoint="$(checkpoint_for_label "${label}")"
  test -d "${checkpoint}"
  sample_args_for_label "${label}"

  local sample_dir="${OUT_DIR}/${label}_sample${SAMPLE_COUNT}"
  local refined_dir="${OUT_DIR}/${label}_refined${REFINE_MAX_PROPOSALS}"
  local child_run_id="${RUN_ID}-${label}-a100"

  activate_conda_env "${MAIN_ENV_NAME}"

  if [ "${SKIP_COMPLETED}" != "1" ] || [ ! -f "${sample_dir}/sample_metrics.json" ]; then
    next_port
    run_logged "${LOG_DIR}/${label}_sample${SAMPLE_COUNT}.log" \
      torchrun --nproc_per_node="${DLM_NPROC}" --master_port="${NEXT_PORT}" scripts/sample_llada_r5_exact_length.py \
        --model-path "${DLM_MODEL_PATH}" \
        --checkpoint-path "${checkpoint}" \
        --prompt-jsonl "${H1A2_PLANS_JSONL}" \
        --output-dir "${sample_dir}" \
        --body-prompt-style full_plan_state \
        --num-samples "${SAMPLE_COUNT}" \
        --batch-size "${DLM_BATCH_SIZE}" \
        --temperature "${DLM_TEMPERATURE}" \
        "${SAMPLE_ARGS[@]}"
  else
    echo "Skipping completed sample for ${label}" | tee -a "${LOG_DIR}/${label}_skip.log"
  fi

  run_logged "${LOG_DIR}/${label}_raw_composition.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${sample_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --text-key text \
      --representation dynamic_v1 \
      --output-json "${NOTES_DIR}/${label}_composition_raw.json" \
      --output-md "${NOTES_DIR}/${label}_composition_raw.md"

  local graph_count
  graph_count="$(python - <<PY
import torch
from pathlib import Path
path = Path("${sample_dir}/proposal_graphs.pt")
graphs = torch.load(path, map_location="cpu") if path.exists() else []
print(len(graphs))
PY
)"
  echo "${graph_count}" > "${NOTES_DIR}/${label}_graph_count.txt"

  if [ "${SKIP_COMPLETED}" != "1" ] || ! find "${refined_dir}" -maxdepth 1 -type f -name 'dlm_refined_mp_*.pt' ! -name '*.rank*.pt' | grep -q .; then
    next_port
    run_logged "${LOG_DIR}/${label}_refine${REFINE_MAX_PROPOSALS}.log" \
      torchrun --nproc_per_node="${REFINE_NPROC}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
        --proposal-graphs "${sample_dir}/proposal_graphs.pt" \
        --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
        --output-dir "${refined_dir}" \
        --max-proposals "${REFINE_MAX_PROPOSALS}" \
        --diff-steps "${DIFF_STEPS}"
  else
    echo "Skipping completed refine for ${label}" | tee -a "${LOG_DIR}/${label}_skip.log"
  fi

  local refined_pt
  refined_pt="$(python - <<PY
from pathlib import Path
candidates = [p for p in Path("${refined_dir}").glob("dlm_refined_mp_*.pt") if ".rank" not in p.name]
if not candidates:
    raise SystemExit("missing refined pt")
print(sorted(candidates, key=lambda p: (p.stat().st_size, p.name), reverse=True)[0])
PY
)"
  echo "${refined_pt}" > "${NOTES_DIR}/${label}_refined_pt.txt"

  run_logged "${LOG_DIR}/${label}_crysllmgen_metrics${REFINE_MAX_PROPOSALS}.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${refined_dir}" \
      --output-json "${NOTES_DIR}/${label}_crysllmgen_metrics${REFINE_MAX_PROPOSALS}.json"

  run_logged "${LOG_DIR}/${label}_composition_refined.log" \
    python scripts/analyze_composition_validity.py \
      --raw-pt "${sample_dir}/raw_dlm_samples.pt" \
      --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
      --text-key text \
      --refined-pt "${refined_pt}" \
      --representation dynamic_v1 \
      --refined-world-size "${REFINE_NPROC}" \
      --output-json "${NOTES_DIR}/${label}_composition${REFINE_MAX_PROPOSALS}.json" \
      --output-md "${NOTES_DIR}/${label}_composition${REFINE_MAX_PROPOSALS}.md"

  run_logged "${LOG_DIR}/${label}_refined_gate.log" \
    python scripts/evaluate_r5b_gate.py \
      --mode refined1000 \
      --sample-metrics "${sample_dir}/sample_metrics.json" \
      --composition-summary "${NOTES_DIR}/${label}_composition${REFINE_MAX_PROPOSALS}.json" \
      --crysllmgen-metrics "${NOTES_DIR}/${label}_crysllmgen_metrics${REFINE_MAX_PROPOSALS}.json" \
      --output-json "${NOTES_DIR}/${label}_refined${REFINE_MAX_PROPOSALS}_gate.json" || true

  activate_conda_env "${A100_ENV_NAME}"
  run_logged "${LOG_DIR}/${label}_a100_cache_missing_pre.log" \
    python scripts/a800/check_a100_eval_sun_cache_missing.py \
      --eval-dir reference/a100_eval_sun \
      --train-csv reference/crysllmgen/data/mp_20/train.csv \
      --cache-path "${MP_CACHE_PATH}" \
      --run "dlm=${refined_pt}" \
      --summary-json "${NOTES_DIR}/${label}_a100_cache_missing_pre.json"

  local missing_count
  missing_count="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/${label}_a100_cache_missing_pre.json").read_text())
run = payload["runs"]["dlm"]
print(int(run.get("missing_chemsys") or 0) + int(run.get("missing_structures") or 0))
PY
)"
  if [ "${missing_count}" != "0" ]; then
    if [ -n "${MP_API_KEY_FILE}" ] && [ -s "${MP_API_KEY_FILE}" ]; then
      run_logged "${LOG_DIR}/${label}_a100_cache_enrich.log" \
        python scripts/a800/enrich_a100_eval_sun_mp_cache.py \
          --eval-dir reference/a100_eval_sun \
          --gen-file "${refined_pt}" \
          --train-csv reference/crysllmgen/data/mp_20/train.csv \
          --cache-path "${MP_CACHE_PATH}" \
          --key-file "${MP_API_KEY_FILE}" \
          --summary-json "${NOTES_DIR}/${label}_a100_cache_enrich.json"
    elif [ "${ALLOW_MISSING_CACHE}" != "1" ]; then
      echo "A100 cache missing ${missing_count} entries for ${label}, and no key file was provided." >&2
      exit 2
    fi
  fi

  run_logged "${LOG_DIR}/${label}_a100_sun.log" \
    env RUN_ID="${child_run_id}" DLM_PT="${refined_pt}" PYTHON_BIN=python \
      MP_CACHE_PATH="${MP_CACHE_PATH}" GLOBAL_RELAX_CACHE_PATH="${GLOBAL_RELAX_CACHE_PATH}" \
      ALLOW_MISSING_CACHE="${ALLOW_MISSING_CACHE}" \
      bash scripts/a800/run_a100_eval_sun_dlm_only.sh

  cp "runs/${child_run_id}/notes/a100_eval_sun_dlm_only_summary.json" "${NOTES_DIR}/${label}_a100_eval_sun_dlm_only_summary.json"
  cp "runs/${child_run_id}/notes/dlm_a100_eval_sun_strict_summary.md" "${NOTES_DIR}/${label}_a100_strict_summary.md"
  cp "runs/${child_run_id}/notes/dlm_a100_eval_sun_meta_like_summary.md" "${NOTES_DIR}/${label}_a100_meta_like_summary.md"
}

for label in ${LABELS}; do
  run_one_label "${label}"
done

export RUN_ID NOTES_DIR OUT_DIR LABELS REFINE_MAX_PROPOSALS SAMPLE_COUNT
python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_DIR"])
out = Path(os.environ["OUT_DIR"])
labels = os.environ["LABELS"].split()
target = os.environ["REFINE_MAX_PROPOSALS"]
sample_count = os.environ["SAMPLE_COUNT"]

def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

rows = []
for label in labels:
    sample = read_json(out / f"{label}_sample{sample_count}" / "sample_metrics.json")
    crys = read_json(notes / f"{label}_crysllmgen_metrics{target}.json")
    comp = read_json(notes / f"{label}_composition{target}.json")
    sun = read_json(notes / f"{label}_a100_eval_sun_dlm_only_summary.json")
    graph_count = None
    graph_path = notes / f"{label}_graph_count.txt"
    if graph_path.exists():
        graph_count = int(graph_path.read_text().strip())
    row = {
        "label": label,
        "sample_metrics": sample,
        "graph_count": graph_count,
        "crysllmgen_metrics": crys,
        "composition": comp,
        "a100": sun,
        "refined_pt": (notes / f"{label}_refined_pt.txt").read_text().strip() if (notes / f"{label}_refined_pt.txt").exists() else None,
    }
    rows.append(row)

payload = {"run_id": os.environ["RUN_ID"], "labels": rows}
(notes / "h1_free_geometry_full1000_remaining_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# H1 Free-Geometry Full1000 Remaining Branches",
    "",
    f"- RUN_ID: `{os.environ['RUN_ID']}`",
    f"- sample_count: `{sample_count}`",
    f"- refine_max_proposals: `{target}`",
    "",
    "| label | graph success | Crys comp_valid | Crys cov_recall | strict adjusted | meta-like adjusted | Novel+Unique | hull eval |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in rows:
    sample = row.get("sample_metrics") or {}
    crys_payload = row.get("crysllmgen_metrics") or {}
    crys = crys_payload.get("metrics", crys_payload) if isinstance(crys_payload, dict) else {}
    sun = row.get("a100") or {}
    strict = sun.get("dlm_strict", {}) if isinstance(sun, dict) else {}
    meta = sun.get("dlm_meta_like", {}) if isinstance(sun, dict) else {}
    lines.append(
        "| {label} | {graph} | {comp} | {cov} | {strict} | {meta} | {nu} | {hull} |".format(
            label=row["label"],
            graph=sample.get("graph_success"),
            comp=crys.get("comp_valid"),
            cov=crys.get("cov_recall"),
            strict=strict.get("coverage-adjusted_sun_estimate_pct"),
            meta=meta.get("coverage-adjusted_sun_estimate_pct"),
            nu=strict.get("novel_+_unique_pct"),
            hull=strict.get("e_hull_evaluated"),
        )
    )
(notes / "h1_free_geometry_full1000_remaining_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
BASELINE_PT="${BASELINE_PT:-/public/home/jiaosz/hengzhang/crysllmgen-main/experiments/diff_guided_llm/baseline_5epoch_uncond_1000/M1_mp_20_1000.pt}"
DLM_PT="${DLM_PT:-runs/20260531_0040-r5c-full1000-sun/outputs/r5c_refined1000/dlm_refined_mp_1000.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
REFERENCE_CSV_DIR="${REFERENCE_CSV_DIR:-reference/crysllmgen/data/mp_20}"
RELAX_MAX_STEPS="${RELAX_MAX_STEPS:-500}"
MAX_NATOMS_PER_BATCH="${MAX_NATOMS_PER_BATCH:-512}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
OUT_DIR="${RUN_DIR}/outputs"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
BASE_DIR="${OUT_DIR}/crysllmgen_baseline"
DLM_DIR="${OUT_DIR}/r5c_dlm"
mkdir -p "${BASE_DIR}" "${DLM_DIR}" "${NOTES_DIR}" "${LOG_DIR}"

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

write_config() {
  python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "stage": "crysllmgen_baseline_vs_r5c_dlm_sun_distribution_compare",
    "baseline_pt": os.environ["BASELINE_PT"],
    "dlm_pt": os.environ["DLM_PT"],
    "reference_dataset": os.environ["REFERENCE_DATASET"],
    "reference_csv_dir": os.environ["REFERENCE_CSV_DIR"],
    "mattergen_root": os.environ["MATTERGEN_ROOT"],
    "mattergen_python": os.environ["MATTERGEN_PYTHON"],
    "mattersim_checkpoint": os.environ["MATTERSIM_CHECKPOINT"],
    "relax_max_steps": int(os.environ["RELAX_MAX_STEPS"]),
    "max_natoms_per_batch": int(os.environ["MAX_NATOMS_PER_BATCH"]),
    "notes": (
        "Baseline is CrysLLMGen unconditional 1000 pt. "
        "DLM is R5-C conditional exact-length refined1000 pt."
    ),
}
Path(os.environ["NOTES_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NOTES_DIR"], "compare_run_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

export RUN_ID BASELINE_PT DLM_PT REFERENCE_DATASET REFERENCE_CSV_DIR MATTERGEN_ROOT
export MATTERGEN_PYTHON MATTERSIM_CHECKPOINT RELAX_MAX_STEPS MAX_NATOMS_PER_BATCH NOTES_DIR

test -f "${BASELINE_PT}"
test -f "${DLM_PT}"
write_config

run_logged "${LOG_DIR}/preflight_py_compile.log" \
  python -m py_compile \
    scripts/run_crysllmgen_metrics.py \
    scripts/analyze_composition_validity.py \
    scripts/analyze_crystal_distribution.py \
    scripts/convert_crysllmgen_pt_to_extxyz.py \
    scripts/run_mattergen_sun_eval.py \
    scripts/analyze_mattergen_sun_detailed.py

for label in baseline dlm; do
  if [ "${label}" = "baseline" ]; then
    PT="${BASELINE_PT}"
    CUR_DIR="${BASE_DIR}"
    PREFIX="baseline"
  else
    PT="${DLM_PT}"
    CUR_DIR="${DLM_DIR}"
    PREFIX="dlm"
  fi

  run_logged "${LOG_DIR}/${PREFIX}_pt_shape.log" \
    python - "${PT}" "${NOTES_DIR}/${PREFIX}_pt_shape.json" <<'PY'
import json
import sys
import torch
from pathlib import Path

pt = Path(sys.argv[1])
out = Path(sys.argv[2])
payload = torch.load(pt, map_location="cpu")
summary = {"path": str(pt), "size_bytes": pt.stat().st_size, "keys": sorted(payload)}
for key, value in payload.items():
    if hasattr(value, "shape"):
        summary[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min().item()) if value.numel() else None,
            "max": float(value.max().item()) if value.numel() else None,
        }
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

  run_logged "${LOG_DIR}/${PREFIX}_crysllmgen_metrics.log" \
    python scripts/run_crysllmgen_metrics.py \
      --root-path "${PT}" \
      --gt-file "${REFERENCE_CSV_DIR}/test.csv" \
      --output-json "${NOTES_DIR}/${PREFIX}_crysllmgen_metrics1000.json"

  run_logged "${LOG_DIR}/${PREFIX}_composition.log" \
    python scripts/analyze_composition_validity.py \
      --refined-pt "${PT}" \
      --reference-csv-dir "${REFERENCE_CSV_DIR}" \
      --output-json "${NOTES_DIR}/${PREFIX}_composition1000.json" \
      --output-md "${NOTES_DIR}/${PREFIX}_composition1000.md"

  run_logged "${LOG_DIR}/${PREFIX}_distribution_test.log" \
    python scripts/analyze_crystal_distribution.py \
      --generated-pt "${PT}" \
      --reference-csv "${REFERENCE_CSV_DIR}/test.csv" \
      --output-json "${NOTES_DIR}/${PREFIX}_distribution_vs_test.json" \
      --output-md "${NOTES_DIR}/${PREFIX}_distribution_vs_test.md"

  if [ -f "${REFERENCE_CSV_DIR}/train.csv" ]; then
    run_logged "${LOG_DIR}/${PREFIX}_distribution_train.log" \
      python scripts/analyze_crystal_distribution.py \
        --generated-pt "${PT}" \
        --reference-csv "${REFERENCE_CSV_DIR}/train.csv" \
        --output-json "${NOTES_DIR}/${PREFIX}_distribution_vs_train.json" \
        --output-md "${NOTES_DIR}/${PREFIX}_distribution_vs_train.md"
  fi

  run_logged "${LOG_DIR}/${PREFIX}_convert_extxyz.log" \
    python scripts/convert_crysllmgen_pt_to_extxyz.py \
      --input-pt "${PT}" \
      --output-extxyz "${CUR_DIR}/generated.extxyz"

  sun_args=(
    --structures-path "${CUR_DIR}/generated.extxyz"
    --reference-dataset "${REFERENCE_DATASET}"
    --save-as "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_metrics.json"
    --save-detailed-as "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_detailed_metrics.json"
    --structures-output-path "${CUR_DIR}/relaxed.extxyz"
    --summary-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_summary.json"
    --relax-failures-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_relax_failures.json"
    --unsupported-failures-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_unsupported_failures.json"
    --metric-errors-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_metric_errors.json"
    --relax-max-steps "${RELAX_MAX_STEPS}"
    --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH}"
    --device cuda
    --structure-matcher disordered
  )
  if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
    sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
  fi

  run_logged "${LOG_DIR}/${PREFIX}_mattergen_sun1000.log" \
    "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

  run_logged "${LOG_DIR}/${PREFIX}_mattergen_sun1000_thresholds.log" \
    python scripts/analyze_mattergen_sun_detailed.py \
      --summary-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_summary.json" \
      --detailed-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_detailed_metrics.json" \
      --label "${PREFIX}_${RUN_ID}" \
      --output-json "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_threshold_analysis.json" \
      --output-md "${NOTES_DIR}/${PREFIX}_mattergen_sun1000_threshold_analysis.md"
done

python - <<'PY'
import json
import math
import os
from collections import Counter
from pathlib import Path

notes = Path(os.environ["RUN_DIR"]) / "notes"

def read(name):
    path = notes / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def metric_value(payload, key):
    value = payload.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value

def comp_summary(label):
    data = read(f"{label}_composition1000.json").get("refined_pt", {})
    reasons = data.get("reason_counts", {})
    count = data.get("count") or 0
    return {
        "count": count,
        "comp_valid": data.get("comp_valid_rate"),
        "single_element": reasons.get("single_element_shortcut", 0) / max(1, count),
        "all_metal": reasons.get("all_metal_shortcut", 0) / max(1, count),
        "charge_neutral_pauling_valid": reasons.get("charge_neutral_pauling_valid", 0) / max(1, count),
        "pbc_duplicate": data.get("pbc_equivalent_duplicate_fraction"),
        "num_atoms_histogram": data.get("num_atoms_histogram", {}),
        "num_elements_histogram": data.get("num_elements_histogram", {}),
        "formula_top30": data.get("formula_top30", {}),
        "reason_counts": reasons,
    }

def crys_summary(label):
    return read(f"{label}_crysllmgen_metrics1000.json").get("metrics", {})

def sun_summary(label):
    threshold = read(f"{label}_mattergen_sun1000_threshold_analysis.json")
    summary = read(f"{label}_mattergen_sun1000_summary.json")
    metrics = read(f"{label}_mattergen_sun1000_metrics.json")
    return {
        "counts": threshold.get("counts", {}),
        "rates_submitted": threshold.get("rates_submitted", {}),
        "rates_successful": threshold.get("rates_successful", {}),
        "ehull_quantiles": threshold.get("ehull_quantiles", {}),
        "mattergen_builtin": {
            key: metric_value(metrics, key)
            for key in [
                "frac_novel_unique_stable_structures",
                "frac_stable_structures",
                "frac_successful_jobs",
                "avg_comp_validity",
                "avg_structure_validity",
                "avg_energy_above_hull_per_atom",
                "frac_novel_unique_structures",
                "frac_unique_structures",
                "precision",
                "recall",
            ]
        },
        "summary_num_structures": summary.get("num_structures"),
        "unsupported_failed": summary.get("n_unsupported_failed"),
        "relax_failed": summary.get("n_relax_failed"),
    }

def dist_summary(label):
    data = read(f"{label}_distribution_vs_test.json")
    gen = data.get("generated", {})
    cmp = data.get("comparison", {})
    return {
        "count": gen.get("count"),
        "density": gen.get("density"),
        "volume_per_atom": gen.get("volume_per_atom"),
        "num_atoms": gen.get("num_atoms"),
        "num_elements": gen.get("num_elements"),
        "high_symmetry_coord_fraction": gen.get("high_symmetry_coord_fraction"),
        "records_all_angles_90": gen.get("records_all_angles_90"),
        "records_all_lengths_equal": gen.get("records_all_lengths_equal"),
        "records_with_exact_duplicate_sites": gen.get("records_with_exact_duplicate_sites"),
        "records_with_same_species_duplicate_sites": gen.get("records_with_same_species_duplicate_sites"),
        "atom_count_histogram": gen.get("atom_count_histogram", {}),
        "num_element_histogram": gen.get("num_element_histogram", {}),
        "element_histogram_top30": gen.get("element_histogram_top30", {}),
        "vs_test_wasserstein": cmp,
    }

def top_mass(counter, k=5):
    total = sum(int(v) for v in counter.values())
    if not total:
        return None
    return sum(sorted((int(v) for v in counter.values()), reverse=True)[:k]) / total

payload = {
    "run_id": os.environ["RUN_ID"],
    "baseline": {
        "pt_shape": read("baseline_pt_shape.json"),
        "crysllmgen": crys_summary("baseline"),
        "composition": comp_summary("baseline"),
        "distribution": dist_summary("baseline"),
        "sun": sun_summary("baseline"),
    },
    "dlm": {
        "pt_shape": read("dlm_pt_shape.json"),
        "crysllmgen": crys_summary("dlm"),
        "composition": comp_summary("dlm"),
        "distribution": dist_summary("dlm"),
        "sun": sun_summary("dlm"),
    },
}

for label in ["baseline", "dlm"]:
    comp = payload[label]["composition"]
    dist = payload[label]["distribution"]
    sun = payload[label]["sun"]
    counts = sun["counts"]
    submitted = counts.get("submitted") or sun.get("summary_num_structures") or 1000
    meta_sun = counts.get("meta_sun", 0)
    strict_sun = counts.get("strict_sun", 0)
    novel_unique = counts.get("novel_unique", 0)
    meta_stable = counts.get("meta_stable", 0)
    payload[label]["derived"] = {
        "meta_sun_submitted": meta_sun / max(1, submitted),
        "strict_sun_submitted": strict_sun / max(1, submitted),
        "p_meta_given_novel_unique": meta_sun / max(1, novel_unique),
        "p_strict_given_novel_unique": strict_sun / max(1, novel_unique),
        "p_novel_unique_given_meta_stable": meta_sun / max(1, meta_stable),
        "top5_formula_mass": top_mass(comp.get("formula_top30", {}), 5),
        "top5_chemsys_or_element_mass_proxy": top_mass(dist.get("element_histogram_top30", {}), 5),
    }

notes.joinpath("comparison_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

def pct(x):
    return "n/a" if x is None else f"{100*float(x):.2f}%"

lines = [
    "# CrysLLMGen Baseline vs R5-C DLM Comparison",
    "",
    "| metric | CrysLLMGen baseline | R5-C DLM |",
    "| --- | ---: | ---: |",
]
for key, label in [
    ("meta_sun_submitted", "meta S.U.N."),
    ("strict_sun_submitted", "strict S.U.N."),
    ("p_meta_given_novel_unique", "P(meta-stable | novel_unique)"),
    ("p_novel_unique_given_meta_stable", "P(novel_unique | meta-stable)"),
]:
    lines.append(f"| {label} | {pct(payload['baseline']['derived'].get(key))} | {pct(payload['dlm']['derived'].get(key))} |")
for key, label in [
    ("comp_valid", "composition valid"),
    ("all_metal", "all metal"),
    ("single_element", "single element"),
]:
    lines.append(f"| {label} | {pct(payload['baseline']['composition'].get(key))} | {pct(payload['dlm']['composition'].get(key))} |")
lines.append("")
lines.append("## Baseline SUN")
lines.append(json.dumps(payload["baseline"]["sun"]["counts"], indent=2, sort_keys=True))
lines.append("")
lines.append("## DLM SUN")
lines.append(json.dumps(payload["dlm"]["sun"]["counts"], indent=2, sort_keys=True))
notes.joinpath("comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:-20260529_r4_e4a_geometry_inpaint}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/llm_grpo_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
R2_CHECKPOINT="${R2_CHECKPOINT:-runs/20260527_semalign_selfimprove_r2/outputs/stage_b/final}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GPU_COUNT="${GPU_COUNT:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
INPAINT_BATCH_SIZE="${INPAINT_BATCH_SIZE:-4}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
ATTEMPTS_PER_SOURCE="${ATTEMPTS_PER_SOURCE:-4}"
RUN_1000_IF_PASS="${RUN_1000_IF_PASS:-1}"
FORCE_1000="${FORCE_1000:-0}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
MASTER_PORT_BASE="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 20000)))}"

if [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be <=2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [ ! -d "${R2_CHECKPOINT}" ]; then
  echo "R2_CHECKPOINT does not exist: ${R2_CHECKPOINT}" >&2
  exit 2
fi

RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${NOTES_DIR}" "${LOG_DIR}" "${OUT_DIR}"

PORT_OFFSET=0
next_port() {
  NEXT_PORT=$((MASTER_PORT_BASE + PORT_OFFSET))
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

python - <<PY
import json
from pathlib import Path
payload = {
  "run_id": "${RUN_ID}",
  "route": "R4 E4A R2 composition-protected geometry inpainting",
  "r2_checkpoint": "${R2_CHECKPOINT}",
  "model_path": "${MODEL_PATH}",
  "temperature": float("${TEMPERATURE}"),
  "gpu_count": int("${GPU_COUNT}"),
  "attempts_per_source": int("${ATTEMPTS_PER_SOURCE}"),
  "no_training": True,
  "no_rl": True,
  "no_10000": True,
}
Path("${NOTES_DIR}/run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_logged "${LOG_DIR}/unit_tests.log" \
  python -m unittest tests.test_diagnostic_remask tests.test_llada_generation_masks tests.test_fixed_slot

analyze_raw_sample() {
  local sample_dir="$1"
  local notes_prefix="$2"
  python scripts/analyze_sample_outputs.py \
    --input-jsonl "${sample_dir}/raw_generations.jsonl" \
    --failure-jsonl "${sample_dir}/failure_cases.jsonl" \
    --output-json "${notes_prefix}_distribution.json" \
    --output-md "${notes_prefix}_distribution.md"
  python scripts/analyze_composition_validity.py \
    --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_prefix}_composition.json" \
    --output-md "${notes_prefix}_composition.md"
  python scripts/analyze_composition_failure_modes.py \
    --raw-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_prefix}_failure_modes.json" \
    --output-md "${notes_prefix}_failure_modes.md" || true
}

sample_r2() {
  local name="$1"
  local sample_dir="$2"
  local num_samples="$3"
  shift 3
  mkdir -p "${sample_dir}"
  if [ -f "${sample_dir}/sample_metrics.json" ]; then
    echo "${name}: found existing ${sample_dir}/sample_metrics.json; reusing sample outputs."
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${name}.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${R2_CHECKPOINT}" \
      --output-dir "${sample_dir}" \
      --num-samples "${num_samples}" \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --block-length 1 \
      --temperature "${TEMPERATURE}" \
      --generation-schedule n-elements-sequential-rest \
      --schema-logit-mask \
      --prefill-slot-tokens \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask \
      "$@"
}

inpaint_one() {
  local name="$1"
  local mode="$2"
  local source_jsonl="$3"
  local output_dir="$4"
  local num_samples="$5"
  shift 5
  mkdir -p "${output_dir}"
  if [ -f "${output_dir}/sample_metrics.json" ]; then
    echo "${name}: found existing ${output_dir}/sample_metrics.json; reusing inpaint outputs."
    return 0
  fi
  next_port
  run_logged "${LOG_DIR}/${name}.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/inpaint_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${R2_CHECKPOINT}" \
      --input-valid-arrays-jsonl "${source_jsonl}" \
      --output-dir "${output_dir}" \
      --mode "${mode}" \
      --num-samples "${num_samples}" \
      --batch-size "${INPAINT_BATCH_SIZE}" \
      --block-length 1 \
      --temperature "${TEMPERATURE}" \
      --schema-logit-mask \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask \
      --anti-high-symmetry \
      --attempts-per-source "${ATTEMPTS_PER_SOURCE}" \
      "$@"
}

BASE256_DIR="${OUT_DIR}/r2_baseline256"
sample_r2 "r2_baseline256_sample" "${BASE256_DIR}" 256
analyze_raw_sample "${BASE256_DIR}" "${NOTES_DIR}/r2_baseline256"

LATTICE256_DIR="${OUT_DIR}/e4a_lattice_only256"
inpaint_one "e4a_lattice_only256" "lattice_only" "${BASE256_DIR}/valid_arrays.jsonl" "${LATTICE256_DIR}" 256 \
  --reject-all-lengths-equal
analyze_raw_sample "${LATTICE256_DIR}" "${NOTES_DIR}/e4a_lattice_only256"
cp "${LATTICE256_DIR}/geometry_diagnostics.json" "${NOTES_DIR}/e4a_lattice_only256_geometry.json" || true
cp "${LATTICE256_DIR}/geometry_diagnostics.md" "${NOTES_DIR}/e4a_lattice_only256_geometry.md" || true

GEOM256_DIR="${OUT_DIR}/e4a_geometry256"
inpaint_one "e4a_geometry256" "geometry" "${BASE256_DIR}/valid_arrays.jsonl" "${GEOM256_DIR}" 256 \
  --reject-all-lengths-equal \
  --max-high-symmetry-coord-fraction 0.75
analyze_raw_sample "${GEOM256_DIR}" "${NOTES_DIR}/e4a_geometry256"
cp "${GEOM256_DIR}/geometry_diagnostics.json" "${NOTES_DIR}/e4a_geometry256_geometry.json" || true
cp "${GEOM256_DIR}/geometry_diagnostics.md" "${NOTES_DIR}/e4a_geometry256_geometry.md" || true

python - <<PY
import json
from pathlib import Path

notes = Path("${NOTES_DIR}")
out = Path("${OUT_DIR}")

def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def comp_summary(name):
    payload = load(notes / f"{name}_composition.json")
    return payload.get("raw_jsonl", payload)

def strict_rate(summary):
    reasons = summary.get("reason_counts") or {}
    count = max(1, int(summary.get("count") or sum(reasons.values()) or 1))
    if "strict_valid_rate" in summary:
        return float(summary["strict_valid_rate"])
    return float(reasons.get("charge_neutral_pauling_valid", 0)) / count

def single_rate(summary):
    reasons = summary.get("reason_counts") or {}
    count = max(1, int(summary.get("count") or sum(reasons.values()) or 1))
    return float(reasons.get("single_element_shortcut", 0)) / count

def geom_summary(name):
    direct = load(notes / f"{name}_geometry.json")
    if direct:
        return {
            "high_symmetry_coord_fraction_mean": direct.get("high_symmetry_coord_fraction_mean"),
            "all_lengths_equal_rate": direct.get("all_lengths_equal_rate"),
            "all_angles_90_rate": direct.get("all_angles_90_rate"),
            "pbc_equivalent_duplicate_count": direct.get("pbc_equivalent_duplicate_count"),
        }
    dist = load(notes / f"{name}_distribution.json")
    total = max(1, int(dist.get("total") or 1))
    return {
        "high_symmetry_coord_fraction_mean": dist.get("high_symmetry_coord_fraction_mean"),
        "all_lengths_equal_rate": float(dist.get("records_all_lengths_equal", 0)) / total,
        "all_angles_90_rate": float(dist.get("records_all_angles_90", 0)) / total,
        "pbc_equivalent_duplicate_count": dist.get("records_with_pbc_equivalent_duplicate_sites", 0),
    }

baseline_comp = comp_summary("r2_baseline256")
baseline_geom = geom_summary("r2_baseline256")
baseline = {
    "comp_valid": float(baseline_comp.get("comp_valid_rate", 0.0)),
    "strict_valid": strict_rate(baseline_comp),
    "single_element": single_rate(baseline_comp),
    **baseline_geom,
}

rows = {}
for name, mode, out_dir in [
    ("e4a_lattice_only256", "lattice_only", out / "e4a_lattice_only256"),
    ("e4a_geometry256", "geometry", out / "e4a_geometry256"),
]:
    sample = load(out_dir / "sample_metrics.json")
    comp = comp_summary(name)
    geom = geom_summary(name)
    row = {
        "mode": mode,
        "output_dir": str(out_dir),
        "parse_rate": float(sample.get("parse_rate", 0.0)),
        "graph_acceptance": float(sample.get("graph_acceptance_rate", sample.get("graph_rate", 0.0))),
        "valid_array_count": int(sample.get("valid_array_count", 0)),
        "comp_valid": float(comp.get("comp_valid_rate", 0.0)),
        "strict_valid": strict_rate(comp),
        "single_element": single_rate(comp),
        **geom,
    }
    failures = []
    if row["parse_rate"] < 0.98:
        failures.append("parse_rate<0.98")
    if row["graph_acceptance"] < 0.95:
        failures.append("graph<0.95")
    if row["comp_valid"] < baseline["comp_valid"] - 0.01:
        failures.append("comp_valid_drop>1pt")
    if row["strict_valid"] < baseline["strict_valid"] - 0.02:
        failures.append("strict_valid_drop>2pt")
    if row["single_element"] > baseline["single_element"] + 1e-9:
        failures.append("single_element_increase")
    if int(row.get("pbc_equivalent_duplicate_count") or 0) != 0:
        failures.append("pbc_duplicate")
    if float(row.get("high_symmetry_coord_fraction_mean") or 1.0) >= 0.55:
        failures.append("high_sym_coord_mean>=0.55")
    if float(row.get("all_lengths_equal_rate") or 1.0) >= 0.50:
        failures.append("a=b=c>=50%")
    if float(row.get("all_angles_90_rate") or 0.0) > float(baseline.get("all_angles_90_rate") or 0.0) + 0.01:
        failures.append("all90_above_baseline")
    row["passed"] = not failures
    row["failures"] = failures
    rows[name] = row

passed = [row for row in rows.values() if row["passed"]]
best = None
if passed:
    best = sorted(
        passed,
        key=lambda row: (
            float(row.get("high_symmetry_coord_fraction_mean") or 1.0),
            float(row.get("all_lengths_equal_rate") or 1.0),
            -float(row.get("comp_valid") or 0.0),
        ),
    )[0]

payload = {
    "baseline": baseline,
    "candidates": rows,
    "passed": bool(best),
    "best_mode": None if best is None else best["mode"],
    "best_output_dir": None if best is None else best["output_dir"],
    "thresholds": {
        "parse_rate": 0.98,
        "graph_acceptance": 0.95,
        "comp_drop_allowed": 0.01,
        "strict_drop_allowed": 0.02,
        "single_element_no_increase": True,
        "high_symmetry_coord_fraction_mean": 0.55,
        "all_lengths_equal_rate": 0.50,
    },
}
(notes / "e4a_256_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

PASSED_256="$(python - <<PY
import json
from pathlib import Path
payload=json.loads((Path("${NOTES_DIR}")/"e4a_256_gate.json").read_text())
print("1" if payload.get("passed") else "0")
PY
)"
if [ "${PASSED_256}" != "1" ] && [ "${FORCE_1000}" != "1" ]; then
  echo "E4A failed 256 gate; stopping before 1000."
  exit 0
fi
if [ "${RUN_1000_IF_PASS}" != "1" ]; then
  echo "RUN_1000_IF_PASS!=1; stopping after 256."
  exit 0
fi

BEST_MODE="$(python - <<PY
import json
from pathlib import Path
payload=json.loads((Path("${NOTES_DIR}")/"e4a_256_gate.json").read_text())
print(payload.get("best_mode") or "geometry")
PY
)"
BEST_EXTRA_ARGS=(--reject-all-lengths-equal)
if [ "${BEST_MODE}" = "geometry" ]; then
  BEST_EXTRA_ARGS+=(--max-high-symmetry-coord-fraction 0.75)
fi

BASE1000_DIR="${OUT_DIR}/r2_baseline1000"
sample_r2 "r2_baseline1000_sample" "${BASE1000_DIR}" "${MAX_ATTEMPTS}" \
  --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
  --max-attempts "${MAX_ATTEMPTS}"
analyze_raw_sample "${BASE1000_DIR}" "${NOTES_DIR}/r2_baseline1000"

TARGET_REACHED="$(python - <<PY
import json
from pathlib import Path
print("1" if json.loads(Path("${BASE1000_DIR}/sample_metrics.json").read_text()).get("target_reached") else "0")
PY
)"
if [ "${TARGET_REACHED}" != "1" ]; then
  echo "R2 baseline1000 source did not reach target graph-valid count; stopping before E4A 1000."
  exit 0
fi

E4A1000_DIR="${OUT_DIR}/e4a_${BEST_MODE}1000"
inpaint_one "e4a_${BEST_MODE}1000" "${BEST_MODE}" "${BASE1000_DIR}/valid_arrays.jsonl" "${E4A1000_DIR}" "${TARGET_GRAPH_SUCCESS}" "${BEST_EXTRA_ARGS[@]}"
analyze_raw_sample "${E4A1000_DIR}" "${NOTES_DIR}/e4a_${BEST_MODE}1000"
cp "${E4A1000_DIR}/geometry_diagnostics.json" "${NOTES_DIR}/e4a_${BEST_MODE}1000_geometry.json" || true
cp "${E4A1000_DIR}/geometry_diagnostics.md" "${NOTES_DIR}/e4a_${BEST_MODE}1000_geometry.md" || true

VALID1000="$(python - <<PY
import json
from pathlib import Path
metrics=json.loads(Path("${E4A1000_DIR}/sample_metrics.json").read_text())
print(int(metrics.get("valid_array_count", 0)))
PY
)"
if [ "${VALID1000}" -lt "${TARGET_GRAPH_SUCCESS}" ]; then
  echo "E4A 1000 produced only ${VALID1000} valid arrays; stopping before refinement/SUN."
  exit 0
fi

REFINED1000_DIR="${OUT_DIR}/refined1000"
SUN1000_DIR="${OUT_DIR}/mattergen_sun1000"
mkdir -p "${REFINED1000_DIR}" "${SUN1000_DIR}"

next_port
run_logged "${LOG_DIR}/refine1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${E4A1000_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED1000_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps "${DIFF_STEPS}" \
    --max-proposals "${TARGET_GRAPH_SUCCESS}"

REFINED_PT="${REFINED1000_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt"
python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED1000_DIR}" \
  --output-json "${NOTES_DIR}/crysllmgen_metrics1000.json"
python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${E4A1000_DIR}/raw_generations.jsonl" \
  --refined-pt "${REFINED_PT}" \
  --refined-world-size 2 \
  --output-json "${NOTES_DIR}/composition1000.json" \
  --output-md "${NOTES_DIR}/composition1000.md"
python scripts/evaluate_mp20_candidate_gate.py \
  --mode refined1000 \
  --sample-metrics "${E4A1000_DIR}/sample_metrics.json" \
  --composition-summary "${NOTES_DIR}/composition1000.json" \
  --composition-key refined_pt \
  --crysllmgen-metrics "${NOTES_DIR}/crysllmgen_metrics1000.json" \
  --max-single-element 0.10 \
  --max-pbc-duplicate 0.0 \
  --output-json "${NOTES_DIR}/refined1000_gate.json" || true

python scripts/convert_crysllmgen_pt_to_extxyz.py \
  --input-pt "${REFINED_PT}" \
  --output-extxyz "${SUN1000_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN1000_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/mattergen_sun1000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json"
  --structures-output-path "${SUN1000_DIR}/relaxed.extxyz"
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json"
  --relax-failures-json "${NOTES_DIR}/mattergen_sun1000_relax_failures.json"
  --unsupported-failures-json "${NOTES_DIR}/mattergen_sun1000_unsupported_failures.json"
  --metric-errors-json "${NOTES_DIR}/mattergen_sun1000_metric_errors.json"
  --relax-max-steps "${RELAX_MAX_STEPS:-500}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi
run_logged "${LOG_DIR}/mattergen_sun1000.log" \
  "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

python scripts/analyze_mattergen_sun_detailed.py \
  --summary-json "${NOTES_DIR}/mattergen_sun1000_summary.json" \
  --detailed-json "${NOTES_DIR}/mattergen_sun1000_detailed_metrics.json" \
  --label "${RUN_ID}" \
  --output-json "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.json" \
  --output-md "${NOTES_DIR}/mattergen_sun1000_threshold_analysis.md"

python scripts/compare_sun_overlap_diagnostics.py \
  --run "R2=runs/20260527_semalign_selfimprove_r2" \
  --run "E4A=${RUN_DIR}" \
  --output-json "${NOTES_DIR}/final_overlap_diagnostics.json" \
  --output-md "${NOTES_DIR}/final_overlap_diagnostics.md" || true

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
def read(name):
    path = notes / name
    return json.loads(path.read_text()) if path.exists() else {}
payload = {
  "run_id": "${RUN_ID}",
  "best_mode": "${BEST_MODE}",
  "e4a_256_gate": read("e4a_256_gate.json"),
  "sample1000": json.loads(Path("${E4A1000_DIR}/sample_metrics.json").read_text()),
  "geometry1000": read("e4a_${BEST_MODE}1000_geometry.json"),
  "crysllmgen": read("crysllmgen_metrics1000.json"),
  "composition1000": read("composition1000.json"),
  "sun_thresholds": read("mattergen_sun1000_threshold_analysis.json"),
  "no_10000": True,
}
(notes / "result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "R4 E4A geometry inpainting run complete: ${RUN_DIR}"

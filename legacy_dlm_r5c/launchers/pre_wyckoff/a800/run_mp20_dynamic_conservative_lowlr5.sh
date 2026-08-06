#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
LOWLR5_CHECKPOINT="${LOWLR5_CHECKPOINT:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_dynamic_v1}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
BRANCH_DIR="${RUN_DIR}/outputs/dynamic_v1_lowlr5"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
REPORT_DIR="reports"
mkdir -p "${BRANCH_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${REPORT_DIR}"

BASE_MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"
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

write_run_config() {
  python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "representation": "dynamic_v1",
    "model_path": "${MODEL_PATH}",
    "start_checkpoint": "${LOWLR5_CHECKPOINT}",
    "data_dir": "${DATA_DIR}",
    "temperature": float("${TEMPERATURE}"),
    "sampling": {
        "block_length": 1,
        "generation_schedule": "dynamic-n-elements-lattice-coords",
        "schema_logit_mask": True,
        "atom_count_grammar_mask": True,
        "duplicate_coordinate_mask": True,
        "lattice_volume_mask": True,
    },
    "gates": {
        "stage_b_min_graph": 0.95,
        "stage_b_min_comp_valid": 0.88,
        "expand_min_parse": 0.98,
        "expand_min_graph": 0.95,
        "expand_min_comp_valid": 0.918,
        "expand_min_strict": 0.57,
    },
}
Path("${NOTES_DIR}/dynamic_v1_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
write_run_config

run_logged "${LOG_DIR}/dynamic_preflight_tests.log" \
  python -m unittest tests.test_dynamic_crystal tests.test_llada_generation_masks tests.test_llada_sft_weights
run_logged "${LOG_DIR}/dynamic_py_compile.log" \
  python -m py_compile crystal_dlm/dynamic_crystal.py crystal_dlm/llada_generation.py scripts/build_dynamic_crystal_sft_data.py scripts/llada_sft.py scripts/sample_llada_dynamic_crystals.py scripts/analyze_composition_validity.py

DATA_READY=$(python - <<PY
import json
from pathlib import Path
data_dir = Path("${DATA_DIR}")
required = [data_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
required += [data_dir / "stats.json"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    print(0)
    raise SystemExit
try:
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
ok = stats.get("representation") == "dynamic_v1"
ok = ok and all(
    int(stats.get("splits", {}).get(split, {}).get("rows_written", 0)) > 0
    for split in ("train", "val", "test")
)
if ok:
    marker_path = data_dir / "_SUCCESS"
    if not marker_path.exists():
        marker_path.write_text(
            json.dumps(
                {
                    "representation": "dynamic_v1",
                    "complete": True,
                    "source": "backfilled_from_stats",
                    "splits": {
                        split: {
                            "rows_seen": stats.get("splits", {}).get(split, {}).get("rows_seen", 0),
                            "rows_written": stats.get("splits", {}).get(split, {}).get("rows_written", 0),
                            "failures": stats.get("splits", {}).get(split, {}).get("failures", 0),
                        }
                        for split in ("train", "val", "test")
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
print(1 if ok else 0)
PY
)

if [[ "${DATA_READY}" != "1" ]]; then
  run_logged "${LOG_DIR}/build_dynamic_data.log" \
    python scripts/build_dynamic_crystal_sft_data.py \
      --input-dir "${INPUT_CSV_DIR}" \
      --output-dir "${DATA_DIR}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --answer-separator ""
else
  echo "Dynamic-v1 data is already complete at ${DATA_DIR}; reusing it." | tee -a "${LOG_DIR}/build_dynamic_data.log"
fi

sample_candidate() {
  local name="$1"
  local checkpoint="$2"
  local out_dir="${BRANCH_DIR}/${name}_sample256"
  local notes_prefix="${NOTES_DIR}/dynamic_${name}"
  if [[ "${REUSE_SAMPLE256:-0}" == "1" \
      && -s "${out_dir}/sample_metrics.json" \
      && -s "${out_dir}/raw_dlm_samples.pt" \
      && -s "${out_dir}/raw_generations.jsonl" \
      && -s "${notes_prefix}_composition256.json" \
      && -s "${notes_prefix}_gate256.json" ]]; then
    echo "Reusing existing ${name} 256 sample outputs at ${out_dir}" | tee -a "${LOG_DIR}/${name}_sample256.log"
  else
    run_logged "${LOG_DIR}/${name}_sample256.log" \
      python scripts/sample_llada_dynamic_crystals.py \
        --model-path "${MODEL_PATH}" \
        --checkpoint-path "${checkpoint}" \
        --output-dir "${out_dir}" \
        --num-samples 256 \
        --batch-size "${SAMPLE_BATCH_SIZE}" \
        --temperature "${TEMPERATURE}"
    run_logged "${LOG_DIR}/${name}_composition256.log" \
      python scripts/analyze_composition_validity.py \
        --raw-pt "${out_dir}/raw_dlm_samples.pt" \
        --raw-generations-jsonl "${out_dir}/raw_generations.jsonl" \
        --representation dynamic_v1 \
        --output-json "${notes_prefix}_composition256.json" \
        --output-md "${notes_prefix}_composition256.md"
    run_logged "${LOG_DIR}/${name}_gate256.log" \
      python scripts/evaluate_mp20_candidate_gate.py \
        --mode smoke256 \
        --sample-metrics "${out_dir}/sample_metrics.json" \
        --composition-summary "${notes_prefix}_composition256.json" \
        --composition-key raw_pt \
        --min-parse-rate 0.98 \
        --min-graph-acceptance 0.95 \
        --max-pbc-duplicate 0.0 \
        --output-json "${notes_prefix}_gate256.json"
  fi
}

train_stage() {
  local name="$1"
  local checkpoint="$2"
  local lr="$3"
  local out_dir="${BRANCH_DIR}/${name}"
  next_port
  run_logged "${LOG_DIR}/${name}_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --data-dir "${DATA_DIR}" \
      --representation dynamic_v1 \
      --output-dir "${out_dir}" \
      --epochs 1 \
      --batch-size "${SFT_BATCH_SIZE}" \
      --grad-accum "${SFT_GRAD_ACCUM}" \
      --lr "${lr}" \
      --lr-scheduler cosine \
      --warmup-steps 100 \
      --min-lr-ratio 0.2 \
      --save-steps 848 \
      --eval-steps 424 \
      --position-diagnostics-steps 848 \
      --atom-count-loss-weight 3.0 \
      --nonempty-slot-loss-weight 3.0 \
      --coordinate-loss-weight 1.0 \
      --empty-slot-loss-weight 0.0 \
      --slot-marker-loss-weight 0.0 \
      --pad-coordinate-loss-weight 0.0
}

if [[ "${REUSE_STAGE_A_FINAL:-0}" == "1" && -s "${BRANCH_DIR}/stage_a_lr2e-5/final/adapter_model.safetensors" ]]; then
  echo "Reusing existing Stage A final checkpoint at ${BRANCH_DIR}/stage_a_lr2e-5/final" | tee -a "${LOG_DIR}/stage_a_lr2e-5_train.log"
else
  train_stage "stage_a_lr2e-5" "${LOWLR5_CHECKPOINT}" "2e-5"
fi
sample_candidate "stage_a_final" "${BRANCH_DIR}/stage_a_lr2e-5/final"

STAGE_A_GATE_JSON="${NOTES_DIR}/dynamic_stage_a_final_gate256.json"
RUN_STAGE_B=$(python - <<PY
import json
from pathlib import Path
gate = json.loads(Path("${STAGE_A_GATE_JSON}").read_text())
m = gate["metrics"]
print(1 if m["graph_acceptance"] >= 0.95 and m["comp_valid"] >= 0.88 else 0)
PY
)

if [[ "${RUN_STAGE_B}" == "1" ]]; then
  train_stage "stage_b_lr8e-6" "${BRANCH_DIR}/stage_a_lr2e-5/final" "8e-6"
  sample_candidate "stage_b_final" "${BRANCH_DIR}/stage_b_lr8e-6/final"
fi

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
candidates = []
for name, checkpoint in [
    ("stage_a_final", "${BRANCH_DIR}/stage_a_lr2e-5/final"),
    ("stage_b_final", "${BRANCH_DIR}/stage_b_lr8e-6/final"),
]:
    gate_path = notes / f"dynamic_{name}_gate256.json"
    comp_path = notes / f"dynamic_{name}_composition256.json"
    sample_path = Path("${BRANCH_DIR}") / f"{name}_sample256" / "sample_metrics.json"
    if not gate_path.exists():
        continue
    gate = json.loads(gate_path.read_text())
    candidates.append({
        "name": name,
        "checkpoint": checkpoint,
        "gate": str(gate_path),
        "composition": str(comp_path),
        "sample_metrics": str(sample_path),
        "metrics": gate["metrics"],
    })
candidates.sort(
    key=lambda item: (
        item["metrics"].get("comp_valid", 0.0),
        item["metrics"].get("strict_valid", 0.0),
        -item["metrics"].get("single_element", 1.0),
    ),
    reverse=True,
)
best = candidates[0] if candidates else None
payload = {"candidates": candidates, "best": best}
(notes / "dynamic_candidate_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# Dynamic-v1 Candidate Selection", ""]
for item in candidates:
    m = item["metrics"]
    lines.append(f"- {item['name']}: comp={m.get('comp_valid', 0):.4f}, strict={m.get('strict_valid', 0):.4f}, graph={m.get('graph_acceptance', 0):.4f}, single={m.get('single_element', 0):.4f}")
if best:
    lines.extend(["", f"Best: `{best['name']}`", f"Checkpoint: `{best['checkpoint']}`"])
(notes / "dynamic_candidate_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

BEST_CHECKPOINT=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/dynamic_candidate_selection.json").read_text()).get("best")
print("" if best is None else best["checkpoint"])
PY
)
EXPAND_TO_1000=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/dynamic_candidate_selection.json").read_text()).get("best")
if not best:
    print(0)
else:
    m = best["metrics"]
    print(1 if m.get("parse_rate", 0) >= 0.98 and m.get("graph_acceptance", 0) >= 0.95 and m.get("comp_valid", 0) >= 0.918 and m.get("strict_valid", 0) >= 0.57 else 0)
PY
)

if [[ "${EXPAND_TO_1000}" != "1" ]]; then
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/dynamic_candidate_selection.json").read_text())
payload["decision"] = "stop_after_256"
payload["reason"] = "best candidate did not reach parse/graph/comp/strict expansion gate"
Path("${NOTES_DIR}/dynamic_final_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit 0
fi

next_port
run_logged "${LOG_DIR}/best_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_dynamic_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --output-dir "${BRANCH_DIR}/best_sample1000" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}"

next_port
run_logged "${LOG_DIR}/best_refine1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${BRANCH_DIR}/best_sample1000/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${BRANCH_DIR}/best_refined1000" \
    --diff-steps 800 \
    --max-proposals 1000

run_logged "${LOG_DIR}/best_crysllmgen_metrics1000.log" \
  python scripts/run_crysllmgen_metrics.py \
    --root-path "${BRANCH_DIR}/best_refined1000" \
    --output-json "${NOTES_DIR}/dynamic_crysllmgen_metrics1000.json"
REFINED_PT=$(find "${BRANCH_DIR}/best_refined1000" -maxdepth 1 -name 'dlm_refined_mp_*.pt' | sort | tail -n 1)
run_logged "${LOG_DIR}/best_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${BRANCH_DIR}/best_sample1000/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${BRANCH_DIR}/best_sample1000/raw_generations.jsonl" \
    --representation dynamic_v1 \
    --refined-pt "${REFINED_PT}" \
    --output-json "${NOTES_DIR}/dynamic_composition1000.json" \
    --output-md "${NOTES_DIR}/dynamic_composition1000.md"
run_logged "${LOG_DIR}/best_gate1000.log" \
  python scripts/evaluate_mp20_candidate_gate.py \
    --mode refined1000 \
    --sample-metrics "${BRANCH_DIR}/best_sample1000/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/dynamic_composition1000.json" \
    --composition-key refined_pt \
    --crysllmgen-metrics "${NOTES_DIR}/dynamic_crysllmgen_metrics1000.json" \
    --output-json "${NOTES_DIR}/dynamic_refined1000_gate.json"

run_logged "${LOG_DIR}/best_convert_extxyz.log" \
  python scripts/convert_crysllmgen_pt_to_extxyz.py \
    --input-pt "${REFINED_PT}" \
    --output-extxyz "${BRANCH_DIR}/best_mattergen_sun1000/generated.extxyz" \
    --max-structures 1000
run_logged "${LOG_DIR}/best_mattergen_sun1000.log" \
  "${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py \
    --structures-path "${BRANCH_DIR}/best_mattergen_sun1000/generated.extxyz" \
    --reference-dataset "${REFERENCE_DATASET}" \
    --save-as "${BRANCH_DIR}/best_mattergen_sun1000/metrics.json" \
    --save-detailed-as "${BRANCH_DIR}/best_mattergen_sun1000/detailed_metrics.json" \
    --structures-output-path "${BRANCH_DIR}/best_mattergen_sun1000/relaxed.extxyz" \
    --summary-json "${NOTES_DIR}/dynamic_mattergen_sun1000_summary.json" \
    --potential-load-path "${MATTERSIM_CHECKPOINT}" \
    --relax-max-steps 500 \
    --relax-fmax 0.05 \
    --max-natoms-per-batch 512 \
    --relax-failures-json "${BRANCH_DIR}/best_mattergen_sun1000/relax_failures.json" \
    --unsupported-failures-json "${BRANCH_DIR}/best_mattergen_sun1000/unsupported_failures.json" \
    --metric-errors-json "${BRANCH_DIR}/best_mattergen_sun1000/metric_errors.json"

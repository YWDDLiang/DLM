#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_PATH="${MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20_cif_lite_modular}"
INPUT_CSV_DIR="${INPUT_CSV_DIR:-reference/crysllmgen/data/mp_20}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
COMPOSITION_GEN_LENGTH="${COMPOSITION_GEN_LENGTH:-96}"
LATTICE_GEN_LENGTH="${LATTICE_GEN_LENGTH:-64}"
SITES_GEN_LENGTH="${SITES_GEN_LENGTH:-384}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
STAGE_A_LR="${STAGE_A_LR:-5e-5}"
STAGE_B_LR="${STAGE_B_LR:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
EVAL_STEPS="${EVAL_STEPS:-1250}"
CIF_LITE_PROMPT_VERSION="${CIF_LITE_PROMPT_VERSION:-cif_lite_modular_v2_strict_schema_prompt}"
CIF_LITE_DATA_SKIP_GRAPH_VALIDATION="${CIF_LITE_DATA_SKIP_GRAPH_VALIDATION:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
RUN_DIR="runs/${RUN_ID}"
BRANCH_DIR="${RUN_DIR}/outputs/cif_lite_modular"
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

run_logged "${LOG_DIR}/cif_lite_preflight_tests.log" \
  python -m unittest tests.test_cif_lite tests.test_llada_sft_weights tests.test_composition_validity
run_logged "${LOG_DIR}/cif_lite_py_compile.log" \
  python -m py_compile \
    crystal_dlm/cif_lite.py \
    scripts/build_cif_lite_sft_data.py \
    scripts/llada_sft.py \
    scripts/sample_llada_cif_lite_modular.py \
    scripts/analyze_composition_validity.py

DATA_READY=$(python - <<PY
import json
from pathlib import Path
data_dir = Path("${DATA_DIR}")
required = [data_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
required += [data_dir / "stats.json", data_dir / "_SUCCESS"]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    print(0)
    raise SystemExit
try:
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
ok = stats.get("representation") == "cif_lite_modular"
ok = ok and stats.get("prompt_version") == "${CIF_LITE_PROMPT_VERSION}"
ok = ok and all(
    int(stats.get("splits", {}).get(split, {}).get("rows_written", 0)) > 0
    for split in ("train", "val", "test")
)
print(1 if ok else 0)
PY
)

if [[ "${DATA_READY}" != "1" ]]; then
  if [[ "${CIF_LITE_DATA_SKIP_GRAPH_VALIDATION}" == "1" ]]; then
    run_logged "${LOG_DIR}/build_cif_lite_data.log" \
      python scripts/build_cif_lite_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${DATA_DIR}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-permute-sites \
        --skip-graph-validation
  else
    run_logged "${LOG_DIR}/build_cif_lite_data.log" \
      python scripts/build_cif_lite_sft_data.py \
        --input-dir "${INPUT_CSV_DIR}" \
        --output-dir "${DATA_DIR}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --train-permute-sites
  fi
else
  echo "CIF-lite modular data is already complete at ${DATA_DIR}; reusing it." | tee -a "${LOG_DIR}/build_cif_lite_data.log"
fi

python - <<PY
import json
from pathlib import Path
stats = json.loads(Path("${DATA_DIR}/stats.json").read_text(encoding="utf-8"))
payload = {
    "run_id": "${RUN_ID}",
    "representation": "cif_lite_modular",
    "model_path": "${MODEL_PATH}",
    "data_dir": "${DATA_DIR}",
    "temperature": float("${TEMPERATURE}"),
    "max_length": int("${MAX_LENGTH}"),
    "stage_a": {"epochs": 3, "lr": float("${STAGE_A_LR}"), "scheduler": "cosine"},
    "stage_b": {"epochs": 2, "lr": float("${STAGE_B_LR}"), "scheduler": "cosine"},
    "data_stats": {
        "max_answer_model_length": stats.get("max_answer_model_length"),
        "max_prompt_model_length": stats.get("max_prompt_model_length"),
        "max_length_recommended": stats.get("max_length_recommended"),
        "train_rows": stats.get("splits", {}).get("train", {}).get("rows_written"),
        "prompt_version": stats.get("prompt_version"),
        "skip_graph_validation_on_rebuild": "${CIF_LITE_DATA_SKIP_GRAPH_VALIDATION}" == "1",
    },
    "sampling": {
        "three_pass": ["composition", "lattice", "sites"],
        "temperature": float("${TEMPERATURE}"),
        "block_length": 1,
        "composition_gen_length": int("${COMPOSITION_GEN_LENGTH}"),
        "lattice_gen_length": int("${LATTICE_GEN_LENGTH}"),
        "sites_gen_length": int("${SITES_GEN_LENGTH}"),
    },
    "gates": {
        "parse_rate": 0.98,
        "graph_acceptance": 0.95,
        "comp_valid": 0.88,
        "strict_valid": 0.40,
        "single_element": 0.10,
        "pbc_duplicate": 0.0,
    },
}
Path("${NOTES_DIR}/cif_lite_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

train_stage() {
  local name="$1"
  local checkpoint="$2"
  local epochs="$3"
  local lr="$4"
  local out_dir="${BRANCH_DIR}/${name}"
  next_port
  if [[ -n "${checkpoint}" ]]; then
    run_logged "${LOG_DIR}/${name}_train.log" \
      torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
        --model-path "${MODEL_PATH}" \
        --checkpoint-path "${checkpoint}" \
        --data-dir "${DATA_DIR}" \
        --representation cif_lite_modular \
        --skip-data-vocab-resize \
        --output-dir "${out_dir}" \
        --max-length "${MAX_LENGTH}" \
        --epochs "${epochs}" \
        --batch-size "${SFT_BATCH_SIZE}" \
        --grad-accum "${SFT_GRAD_ACCUM}" \
        --lr "${lr}" \
        --lr-scheduler cosine \
        --warmup-steps 100 \
        --min-lr-ratio 0.2 \
        --save-steps "${SAVE_STEPS}" \
        --eval-steps "${EVAL_STEPS}" \
        --eval-max-batches 50 \
        --position-diagnostics-steps "${SAVE_STEPS}" \
        --composition-module-loss-weight 2.0 \
        --sites-module-loss-weight 1.25 \
        --lattice-module-loss-weight 1.0 \
        --modules-to-save "" \
        --save-embedding-layers false
  else
    run_logged "${LOG_DIR}/${name}_train.log" \
      torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --representation cif_lite_modular \
        --skip-data-vocab-resize \
        --output-dir "${out_dir}" \
        --max-length "${MAX_LENGTH}" \
        --epochs "${epochs}" \
        --batch-size "${SFT_BATCH_SIZE}" \
        --grad-accum "${SFT_GRAD_ACCUM}" \
        --lr "${lr}" \
        --lr-scheduler cosine \
        --warmup-steps 100 \
        --min-lr-ratio 0.2 \
        --save-steps "${SAVE_STEPS}" \
        --eval-steps "${EVAL_STEPS}" \
        --eval-max-batches 50 \
        --position-diagnostics-steps "${SAVE_STEPS}" \
        --composition-module-loss-weight 2.0 \
        --sites-module-loss-weight 1.25 \
        --lattice-module-loss-weight 1.0 \
        --modules-to-save "" \
        --save-embedding-layers false
  fi
}

sample_candidate() {
  local name="$1"
  local checkpoint="$2"
  local out_dir="${BRANCH_DIR}/${name}_sample256"
  local notes_prefix="${NOTES_DIR}/cif_lite_${name}"
  run_logged "${LOG_DIR}/${name}_sample256.log" \
    python scripts/sample_llada_cif_lite_modular.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint}" \
      --output-dir "${out_dir}" \
      --num-samples 256 \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --composition-gen-length "${COMPOSITION_GEN_LENGTH}" \
      --lattice-gen-length "${LATTICE_GEN_LENGTH}" \
      --sites-gen-length "${SITES_GEN_LENGTH}" \
      --temperature "${TEMPERATURE}"
  local raw_pt_args=()
  local composition_key="raw_jsonl"
  if [[ -s "${out_dir}/raw_dlm_samples.pt" ]]; then
    raw_pt_args=(--raw-pt "${out_dir}/raw_dlm_samples.pt")
    composition_key="raw_pt"
  fi
  run_logged "${LOG_DIR}/${name}_composition256.log" \
    python scripts/analyze_composition_validity.py \
      "${raw_pt_args[@]}" \
      --raw-generations-jsonl "${out_dir}/raw_generations.jsonl" \
      --representation cif_lite_modular \
      --output-json "${notes_prefix}_composition256.json" \
      --output-md "${notes_prefix}_composition256.md"
  run_logged "${LOG_DIR}/${name}_gate256.log" \
    python scripts/evaluate_mp20_candidate_gate.py \
      --mode smoke256 \
      --sample-metrics "${out_dir}/sample_metrics.json" \
      --composition-summary "${notes_prefix}_composition256.json" \
      --composition-key "${composition_key}" \
      --min-parse-rate 0.98 \
      --min-graph-acceptance 0.95 \
      --min-comp-valid 0.88 \
      --min-strict-valid 0.40 \
      --max-single-element 0.10 \
      --max-pbc-duplicate 0.0 \
      --output-json "${notes_prefix}_gate256.json"
}

collect_stage_candidates() {
  local stage_name="$1"
  local stage_dir="${BRANCH_DIR}/${stage_name}"
  local candidates_file="${NOTES_DIR}/cif_lite_${stage_name}_candidates.txt"
  : > "${candidates_file}"
  if [[ -d "${stage_dir}/checkpoints" ]]; then
    find "${stage_dir}/checkpoints" -maxdepth 1 -mindepth 1 -type d -name 'step-*' | sort >> "${candidates_file}"
  fi
  if [[ -d "${stage_dir}/final" ]]; then
    echo "${stage_dir}/final" >> "${candidates_file}"
  fi
}

sample_stage_candidates() {
  local stage_name="$1"
  local candidates_file="${NOTES_DIR}/cif_lite_${stage_name}_candidates.txt"
  while IFS= read -r checkpoint; do
    [[ -n "${checkpoint}" ]] || continue
    local base_name
    base_name="$(basename "${checkpoint}")"
    if [[ "${base_name}" == "final" ]]; then
      base_name="${stage_name}_final"
    else
      base_name="${stage_name}_${base_name}"
    fi
    sample_candidate "${base_name}" "${checkpoint}"
  done < "${candidates_file}"
}

select_best_candidate() {
  python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
branch = Path("${BRANCH_DIR}")
candidates = []
for gate_path in sorted(notes.glob("cif_lite_*_gate256.json")):
    name = gate_path.name[len("cif_lite_"):-len("_gate256.json")]
    comp_path = notes / f"cif_lite_{name}_composition256.json"
    sample_path = branch / f"{name}_sample256" / "sample_metrics.json"
    checkpoint = None
    if "_step-" in name:
        stage, step = name.split("_step-", 1)
        checkpoint = str(branch / stage / "checkpoints" / f"step-{step}")
    elif name.endswith("_final"):
        stage = name[:-len("_final")]
        checkpoint = str(branch / stage / "final")
    if checkpoint is None or not Path(checkpoint).exists():
        continue
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    metrics = gate.get("metrics", {})
    candidates.append({
        "name": name,
        "checkpoint": checkpoint,
        "gate": str(gate_path),
        "composition": str(comp_path),
        "sample_metrics": str(sample_path),
        "passed": bool(gate.get("passed")),
        "metrics": metrics,
    })
valid = [
    item for item in candidates
    if item["metrics"].get("parse_rate", 0) >= 0.98 and item["metrics"].get("graph_acceptance", 0) >= 0.95
]
pool = valid or candidates
pool.sort(
    key=lambda item: (
        item["metrics"].get("comp_valid", 0.0),
        item["metrics"].get("strict_valid", 0.0),
        -item["metrics"].get("single_element", 1.0),
    ),
    reverse=True,
)
best = pool[0] if pool else None
payload = {"candidates": candidates, "best": best}
(notes / "cif_lite_candidate_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# CIF-lite Modular Candidate Selection", ""]
for item in candidates:
    m = item["metrics"]
    lines.append(f"- {item['name']}: pass={item['passed']}, comp={m.get('comp_valid', 0):.4f}, strict={m.get('strict_valid', 0):.4f}, graph={m.get('graph_acceptance', 0):.4f}, parse={m.get('parse_rate', 0):.4f}, single={m.get('single_element', 0):.4f}")
if best:
    lines.extend(["", f"Best: `{best['name']}`", f"Checkpoint: `{best['checkpoint']}`"])
(notes / "cif_lite_candidate_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

cleanup_stage_checkpoints() {
  python - <<PY
import json
import shutil
from pathlib import Path
notes = Path("${NOTES_DIR}")
branch = Path("${BRANCH_DIR}")
best = json.loads((notes / "cif_lite_candidate_selection.json").read_text(encoding="utf-8")).get("best")
keep = set()
if best:
    keep.add(str(Path(best["checkpoint"]).resolve()))
for final_dir in branch.glob("stage_*/final"):
    keep.add(str(final_dir.resolve()))
deleted = []
for ckpt in branch.glob("stage_*/checkpoints/step-*"):
    resolved = str(ckpt.resolve())
    if resolved in keep:
        continue
    has_weights = any((ckpt / name).exists() for name in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"))
    if has_weights:
        shutil.rmtree(ckpt)
        deleted.append(str(ckpt))
manifest = {"keep": sorted(keep), "deleted": deleted}
(notes / "cif_lite_checkpoint_cleanup_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if [[ "${REUSE_SMOKE:-0}" != "1" || ! -s "${BRANCH_DIR}/stage0_smoke/final/adapter_model.safetensors" ]]; then
  next_port
  run_logged "${LOG_DIR}/stage0_smoke_train.log" \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/llada_sft.py \
      --model-path "${MODEL_PATH}" \
      --data-dir "${DATA_DIR}" \
      --representation cif_lite_modular \
      --skip-data-vocab-resize \
      --output-dir "${BRANCH_DIR}/stage0_smoke" \
      --max-length "${MAX_LENGTH}" \
      --limit-train 32 \
      --limit-val 32 \
      --epochs 1 \
      --batch-size 1 \
      --grad-accum 1 \
      --lr 1e-4 \
      --lr-scheduler cosine \
      --warmup-steps 2 \
      --min-lr-ratio 0.2 \
      --save-steps 32 \
      --eval-steps 16 \
      --composition-module-loss-weight 2.0 \
      --sites-module-loss-weight 1.25 \
      --lattice-module-loss-weight 1.0 \
      --modules-to-save "" \
      --save-embedding-layers false
fi

run_logged "${LOG_DIR}/stage0_sample64.log" \
  python scripts/sample_llada_cif_lite_modular.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BRANCH_DIR}/stage0_smoke/final" \
    --output-dir "${BRANCH_DIR}/stage0_sample64" \
    --num-samples 64 \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --composition-gen-length "${COMPOSITION_GEN_LENGTH}" \
    --lattice-gen-length "${LATTICE_GEN_LENGTH}" \
    --sites-gen-length "${SITES_GEN_LENGTH}" \
    --temperature "${TEMPERATURE}"

if [[ "${REUSE_STAGE_A_FINAL:-0}" != "1" || ! -s "${BRANCH_DIR}/stage_a/final/adapter_model.safetensors" ]]; then
  train_stage "stage_a" "" 3 "${STAGE_A_LR}"
fi
collect_stage_candidates "stage_a"
sample_stage_candidates "stage_a"
select_best_candidate
cleanup_stage_checkpoints

RUN_STAGE_B=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/cif_lite_candidate_selection.json").read_text()).get("best")
if not best:
    print(0)
else:
    m = best["metrics"]
    print(1 if m.get("parse_rate", 0) >= 0.98 and m.get("graph_acceptance", 0) >= 0.95 and m.get("comp_valid", 0) >= 0.88 and m.get("strict_valid", 0) >= 0.40 and m.get("single_element", 1) <= 0.10 and m.get("pbc_duplicate", 1) <= 0.0 else 0)
PY
)

BEST_CHECKPOINT=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/cif_lite_candidate_selection.json").read_text()).get("best")
print("" if best is None else best["checkpoint"])
PY
)

if [[ "${RUN_STAGE_B}" == "1" ]]; then
  if [[ "${REUSE_STAGE_B_FINAL:-0}" != "1" || ! -s "${BRANCH_DIR}/stage_b/final/adapter_model.safetensors" ]]; then
    train_stage "stage_b" "${BEST_CHECKPOINT}" 2 "${STAGE_B_LR}"
  fi
  collect_stage_candidates "stage_b"
  sample_stage_candidates "stage_b"
  select_best_candidate
  cleanup_stage_checkpoints
  BEST_CHECKPOINT=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/cif_lite_candidate_selection.json").read_text()).get("best")
print("" if best is None else best["checkpoint"])
PY
)
fi

EXPAND_TO_1000=$(python - <<PY
import json
from pathlib import Path
best = json.loads(Path("${NOTES_DIR}/cif_lite_candidate_selection.json").read_text()).get("best")
if not best:
    print(0)
else:
    m = best["metrics"]
    print(1 if m.get("parse_rate", 0) >= 0.98 and m.get("graph_acceptance", 0) >= 0.95 and m.get("comp_valid", 0) >= 0.88 and m.get("strict_valid", 0) >= 0.40 and m.get("single_element", 1) <= 0.10 and m.get("pbc_duplicate", 1) <= 0.0 else 0)
PY
)

if [[ "${EXPAND_TO_1000}" != "1" ]]; then
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${NOTES_DIR}/cif_lite_candidate_selection.json").read_text())
payload["decision"] = "stop_after_256"
payload["reason"] = "best CIF-lite candidate did not reach 256 parse/graph/comp/strict/single/PBC gate"
Path("${NOTES_DIR}/cif_lite_final_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit 0
fi

next_port
run_logged "${LOG_DIR}/best_sample1000.log" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${NEXT_PORT}" scripts/sample_llada_cif_lite_modular.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --output-dir "${BRANCH_DIR}/best_sample1000" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --composition-gen-length "${COMPOSITION_GEN_LENGTH}" \
    --lattice-gen-length "${LATTICE_GEN_LENGTH}" \
    --sites-gen-length "${SITES_GEN_LENGTH}" \
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
    --output-json "${NOTES_DIR}/cif_lite_crysllmgen_metrics1000.json"
REFINED_PT=$(find "${BRANCH_DIR}/best_refined1000" -maxdepth 1 -name 'dlm_refined_mp_*.pt' | sort | tail -n 1)
run_logged "${LOG_DIR}/best_composition1000.log" \
  python scripts/analyze_composition_validity.py \
    --raw-pt "${BRANCH_DIR}/best_sample1000/raw_dlm_samples.pt" \
    --raw-generations-jsonl "${BRANCH_DIR}/best_sample1000/raw_generations.jsonl" \
    --representation cif_lite_modular \
    --refined-pt "${REFINED_PT}" \
    --output-json "${NOTES_DIR}/cif_lite_composition1000.json" \
    --output-md "${NOTES_DIR}/cif_lite_composition1000.md"
run_logged "${LOG_DIR}/best_gate1000.log" \
  python scripts/evaluate_mp20_candidate_gate.py \
    --mode refined1000 \
    --sample-metrics "${BRANCH_DIR}/best_sample1000/sample_metrics.json" \
    --composition-summary "${NOTES_DIR}/cif_lite_composition1000.json" \
    --composition-key refined_pt \
    --crysllmgen-metrics "${NOTES_DIR}/cif_lite_crysllmgen_metrics1000.json" \
    --output-json "${NOTES_DIR}/cif_lite_refined1000_gate.json"

python - <<PY
import json
from pathlib import Path
notes = Path("${NOTES_DIR}")
payload = {
    "decision": "completed_refined1000",
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "candidate_selection": json.loads((notes / "cif_lite_candidate_selection.json").read_text(encoding="utf-8")),
    "crysllmgen_metrics1000": json.loads((notes / "cif_lite_crysllmgen_metrics1000.json").read_text(encoding="utf-8")),
    "gate1000": json.loads((notes / "cif_lite_refined1000_gate.json").read_text(encoding="utf-8")),
}
(notes / "cif_lite_final_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

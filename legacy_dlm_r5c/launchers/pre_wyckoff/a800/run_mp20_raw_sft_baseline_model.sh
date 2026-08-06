#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required, e.g. llada8bbase or llada21mini}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
DATA_DIR="${DATA_DIR:-data/dlm_sft/mp_20}"
GENERATION_SCHEDULE="${GENERATION_SCHEDULE:-n-elements-sequential-rest}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
FALLBACK_BATCH_SIZE="${FALLBACK_BATCH_SIZE:-8}"
FALLBACK_GRAD_ACCUM="${FALLBACK_GRAD_ACCUM:-2}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_A_LR="${STAGE_A_LR:-1e-4}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-8}"
STAGE_B_LR="${STAGE_B_LR:-3e-5}"
START_AT_STAGE_B="${START_AT_STAGE_B:-0}"
STAGE_A_FINAL_PATH="${STAGE_A_FINAL_PATH:-}"
SAVE_STEPS="${SAVE_STEPS:-848}"
EVAL_STEPS="${EVAL_STEPS:-424}"
POSITION_DIAGNOSTICS_STEPS="${POSITION_DIAGNOSTICS_STEPS:-848}"
CRYSLLMGEN_CHECKPOINT="${CRYSLLMGEN_CHECKPOINT:-/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt}"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-/public/home/jiaosz/ywliang/ai4s/mattergen}"
MATTERGEN_PYTHON="${MATTERGEN_PYTHON:-/public/home/jiaosz/miniconda3/envs/crysllm_matgen/bin/python}"
REFERENCE_DATASET="${REFERENCE_DATASET:-${MATTERGEN_ROOT}/data-release/alex-mp/reference_MP2020correction.gz}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/public/home/jiaosz/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-128}"
DIFF_STEPS="${DIFF_STEPS:-800}"
TARGET_GRAPH_SUCCESS="${TARGET_GRAPH_SUCCESS:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1800}"
REFINED_WORLD_SIZE="${REFINED_WORLD_SIZE:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
BRANCH_DIR="${RUN_DIR}/outputs/${MODEL_LABEL}"
NOTES_DIR="${RUN_DIR}/notes"
LOG_DIR="${RUN_DIR}/logs"
REPORT_DIR="reports"
mkdir -p "${BRANCH_DIR}" "${NOTES_DIR}" "${LOG_DIR}" "${REPORT_DIR}"
export RUN_ID MODEL_LABEL MODEL_PATH PROJECT_ROOT DATA_DIR GENERATION_SCHEDULE TEMPERATURE
export RUN_DIR BRANCH_DIR NOTES_DIR LOG_DIR REPORT_DIR

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

python - <<PY
import json
from pathlib import Path

model = Path("${MODEL_PATH}")
data = Path("${DATA_DIR}")
status = {
    "run_id": "${RUN_ID}",
    "model_label": "${MODEL_LABEL}",
    "model_path": "${MODEL_PATH}",
    "data_dir": "${DATA_DIR}",
    "preflight_ok": False,
    "errors": [],
}
if not model.exists() or not model.is_dir():
    status["errors"].append("model_dir_missing")
else:
    files = {p.name: p.stat().st_size for p in model.iterdir() if p.is_file()}
    status["model_files"] = files
    if not files:
        status["errors"].append("model_dir_empty")
    if "config.json" not in files:
        status["errors"].append("missing_config_json")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        status["errors"].append("missing_weight_files")
if not data.exists():
    status["errors"].append("data_dir_missing")
else:
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "vocab_tokens.txt", "stats.json"):
        if not (data / name).exists():
            status["errors"].append(f"missing_data_{name}")
status["preflight_ok"] = not status["errors"]
Path("${NOTES_DIR}/${MODEL_LABEL}_preflight.json").write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not status["preflight_ok"]:
    raise SystemExit("preflight_failed:" + ",".join(status["errors"]))
PY

write_run_config() {
  python - <<PY
import json
from pathlib import Path

payload = {
    "run_id": "${RUN_ID}",
    "model_label": "${MODEL_LABEL}",
    "model_path": "${MODEL_PATH}",
    "data_dir": "${DATA_DIR}",
    "generation_schedule": "${GENERATION_SCHEDULE}",
    "temperature": float("${TEMPERATURE}"),
    "sft": {
        "stage_a_epochs": int("${STAGE_A_EPOCHS}"),
        "stage_a_lr": float("${STAGE_A_LR}"),
        "stage_b_epochs": int("${STAGE_B_EPOCHS}"),
        "stage_b_lr": float("${STAGE_B_LR}"),
        "start_at_stage_b": bool(int("${START_AT_STAGE_B}")),
        "stage_a_final_path": "${STAGE_A_FINAL_PATH}",
        "batch_size": int("${SFT_BATCH_SIZE}"),
        "grad_accum": int("${SFT_GRAD_ACCUM}"),
        "fallback_batch_size": int("${FALLBACK_BATCH_SIZE}"),
        "fallback_grad_accum": int("${FALLBACK_GRAD_ACCUM}"),
        "save_steps": int("${SAVE_STEPS}"),
        "eval_steps": int("${EVAL_STEPS}"),
        "train_prefill_slot_tokens": True,
        "loss_weights": {
            "atom_count": 3.0,
            "slot_marker": 0.25,
            "empty": 0.5,
            "nonempty": 2.0,
            "late_nonempty": 4.0,
            "coordinate": 1.0,
            "pad_coordinate": 0.2,
        },
    },
    "sampling": {
        "temperature": float("${TEMPERATURE}"),
        "block_length": 1,
        "generation_schedule": "${GENERATION_SCHEDULE}",
        "schema_logit_mask": True,
        "prefill_slot_tokens": True,
        "atom_count_grammar_mask": True,
        "duplicate_coordinate_mask": True,
        "lattice_volume_mask": True,
    },
    "refinement": {
        "crysllmgen_checkpoint": "${CRYSLLMGEN_CHECKPOINT}",
        "diff_steps": int("${DIFF_STEPS}"),
    },
    "mattergen": {
        "reference_dataset": "${REFERENCE_DATASET}",
        "mattersim_checkpoint": "${MATTERSIM_CHECKPOINT}",
        "meta_sun_ehull_lt": 0.1,
        "strict_sun_ehull_lt": 0.0,
    },
}
Path("${NOTES_DIR}/${MODEL_LABEL}_run_config.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}
write_run_config

prune_checkpoints_to_top1_latest() {
  local stage_dir="$1"
  local manifest_name="$2"
  [ -d "${stage_dir}/checkpoints" ] || return 0
  PRUNE_STAGE_DIR="${stage_dir}" PRUNE_MANIFEST="${NOTES_DIR}/${MODEL_LABEL}_${manifest_name}_checkpoint_prune.json" python - <<'PY'
import json
import os
import re
import shutil
from pathlib import Path

stage_dir = Path(os.environ["PRUNE_STAGE_DIR"])
manifest = Path(os.environ["PRUNE_MANIFEST"])
ckpt_dir = stage_dir / "checkpoints"
step_re = re.compile(r"^step-(\d+)$")

checkpoints = []
for item in ckpt_dir.iterdir() if ckpt_dir.exists() else []:
    if not item.is_dir():
        continue
    match = step_re.match(item.name)
    if match:
        checkpoints.append((int(match.group(1)), item))
checkpoints.sort()

eval_losses = {}
log = stage_dir / "training_log.jsonl"
if log.exists():
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "eval" and "step" in row and "val_loss" in row:
                try:
                    eval_losses[int(row["step"])] = float(row["val_loss"])
                except Exception:
                    pass

best_step = None
if checkpoints:
    scored = [(eval_losses[step], step) for step, _ in checkpoints if step in eval_losses]
    if scored:
        best_step = min(scored)[1]
latest_step = checkpoints[-1][0] if checkpoints else None

keep_steps = set()
if best_step is not None:
    keep_steps.add(best_step)
elif latest_step is not None and not (stage_dir / "final").exists():
    keep_steps.add(latest_step)

removed = []
kept = []
for step, path in checkpoints:
    if step in keep_steps:
        kept.append(str(path))
        continue
    shutil.rmtree(path)
    removed.append(str(path))

payload = {
    "stage_dir": str(stage_dir),
    "best_step": best_step,
    "latest_step": latest_step,
    "final_exists": (stage_dir / "final").exists(),
    "kept_step_checkpoints": kept,
    "removed_step_checkpoints": removed,
    "eval_losses": {str(k): v for k, v in sorted(eval_losses.items())},
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

run_sft_stage() {
  local stage_name="$1"
  local checkpoint_path="$2"
  local epochs="$3"
  local lr="$4"
  local output_dir="${BRANCH_DIR}/${stage_name}"
  local log_file="${LOG_DIR}/${MODEL_LABEL}_${stage_name}.log"
  local port
  next_port
  port="${NEXT_PORT}"

  local cmd=(
    torchrun --nproc_per_node=2 --master_port "${port}" scripts/llada_sft.py
    --model-path "${MODEL_PATH}"
    --data-dir "${DATA_DIR}"
    --output-dir "${output_dir}"
    --epochs "${epochs}"
    --batch-size "${SFT_BATCH_SIZE}"
    --grad-accum "${SFT_GRAD_ACCUM}"
    --lr "${lr}"
    --lr-scheduler cosine
    --warmup-steps 100
    --min-lr-ratio 0.2
    --save-steps "${SAVE_STEPS}"
    --eval-steps "${EVAL_STEPS}"
    --position-diagnostics-steps "${POSITION_DIAGNOSTICS_STEPS}"
    --eval-max-batches 50
    --atom-count-loss-weight 3.0
    --slot-marker-loss-weight 0.25
    --empty-slot-loss-weight 0.5
    --nonempty-slot-loss-weight 2.0
    --late-nonempty-slot-loss-weight 4.0
    --coordinate-loss-weight 1.0
    --pad-coordinate-loss-weight 0.2
    --train-prefill-slot-tokens
  )
  if [ "${stage_name}" = "smoke_sft" ]; then
    cmd+=(--limit-train 32 --limit-val 32 --eval-max-batches 4)
  fi
  if [ -n "${checkpoint_path}" ]; then
    cmd+=(--checkpoint-path "${checkpoint_path}")
  fi

  if run_logged "${log_file}" "${cmd[@]}"; then
    LAST_SFT_OUTPUT="${output_dir}"
    return 0
  fi

  if grep -Eiq "out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED" "${log_file}"; then
    local fallback_dir="${BRANCH_DIR}/${stage_name}_b8ga2"
    local fallback_log="${LOG_DIR}/${MODEL_LABEL}_${stage_name}_b8ga2.log"
    next_port
    port="${NEXT_PORT}"
    cmd=(
      torchrun --nproc_per_node=2 --master_port "${port}" scripts/llada_sft.py
      --model-path "${MODEL_PATH}"
      --data-dir "${DATA_DIR}"
      --output-dir "${fallback_dir}"
      --epochs "${epochs}"
      --batch-size "${FALLBACK_BATCH_SIZE}"
      --grad-accum "${FALLBACK_GRAD_ACCUM}"
      --lr "${lr}"
      --lr-scheduler cosine
      --warmup-steps 100
      --min-lr-ratio 0.2
      --save-steps "${SAVE_STEPS}"
      --eval-steps "${EVAL_STEPS}"
      --position-diagnostics-steps "${POSITION_DIAGNOSTICS_STEPS}"
      --eval-max-batches 50
      --atom-count-loss-weight 3.0
      --slot-marker-loss-weight 0.25
      --empty-slot-loss-weight 0.5
      --nonempty-slot-loss-weight 2.0
      --late-nonempty-slot-loss-weight 4.0
      --coordinate-loss-weight 1.0
      --pad-coordinate-loss-weight 0.2
      --train-prefill-slot-tokens
    )
    if [ "${stage_name}" = "smoke_sft" ]; then
      cmd+=(--limit-train 32 --limit-val 32 --eval-max-batches 4)
    fi
    if [ -n "${checkpoint_path}" ]; then
      cmd+=(--checkpoint-path "${checkpoint_path}")
    fi
    run_logged "${fallback_log}" "${cmd[@]}"
    LAST_SFT_OUTPUT="${fallback_dir}"
    return 0
  fi

  echo "SFT stage ${stage_name} failed for non-OOM reason; see ${log_file}" >&2
  return 1
}

if [ "${START_AT_STAGE_B}" = "1" ]; then
  if [ -z "${STAGE_A_FINAL_PATH}" ] || [ ! -d "${STAGE_A_FINAL_PATH}" ]; then
    echo "START_AT_STAGE_B=1 requires STAGE_A_FINAL_PATH to point at an existing adapter/full checkpoint directory." >&2
    exit 1
  fi
  STAGE_A_DIR="$(dirname "${STAGE_A_FINAL_PATH}")"
  STAGE_A_CHECKPOINT="${STAGE_A_FINAL_PATH}"
  export STAGE_A_DIR
else
  run_sft_stage "smoke_sft" "" 1 "${STAGE_A_LR}"
  SMOKE_DIR="${LAST_SFT_OUTPUT}"

  run_sft_stage "stage_a_fast${STAGE_A_EPOCHS}e" "" "${STAGE_A_EPOCHS}" "${STAGE_A_LR}"
  STAGE_A_DIR="${LAST_SFT_OUTPUT}"
  STAGE_A_CHECKPOINT="${STAGE_A_DIR}/final"
  export STAGE_A_DIR
  prune_checkpoints_to_top1_latest "${STAGE_A_DIR}" "stage_a"
fi

run_sft_stage "stage_b_lr${STAGE_B_LR}_${STAGE_B_EPOCHS}e" "${STAGE_A_CHECKPOINT}" "${STAGE_B_EPOCHS}" "${STAGE_B_LR}"
STAGE_B_DIR="${LAST_SFT_OUTPUT}"
export STAGE_B_DIR
prune_checkpoints_to_top1_latest "${STAGE_B_DIR}" "stage_b"

CANDIDATE_MANIFEST="${NOTES_DIR}/${MODEL_LABEL}_candidate_manifest.tsv"
export CANDIDATE_MANIFEST
python - <<'PY'
import json
import os
import re
import shutil
from pathlib import Path

notes_dir = Path(os.environ["NOTES_DIR"])
stage_a = Path(os.environ["STAGE_A_DIR"])
stage_b = Path(os.environ["STAGE_B_DIR"])
manifest = Path(os.environ["CANDIDATE_MANIFEST"])
label = os.environ["MODEL_LABEL"]
step_re = re.compile(r"^step-(\d+)$")


def eval_losses(stage_dir: Path) -> dict[int, float]:
    out = {}
    log = stage_dir / "training_log.jsonl"
    if not log.exists():
        return out
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "eval" and "step" in row and "val_loss" in row:
                try:
                    out[int(row["step"])] = float(row["val_loss"])
                except Exception:
                    pass
    return out


records = []
for stage_name, stage_dir in (("stage_a", stage_a), ("stage_b", stage_b)):
    losses = eval_losses(stage_dir)
    ckpt_dir = stage_dir / "checkpoints"
    if not ckpt_dir.exists():
        continue
    for item in ckpt_dir.iterdir():
        if not item.is_dir():
            continue
        match = step_re.match(item.name)
        if not match:
            continue
        step = int(match.group(1))
        records.append(
            {
                "name": f"{stage_name}_{item.name}",
                "path": str(item),
                "stage": stage_name,
                "step": step,
                "val_loss": losses.get(step),
            }
        )

top1 = None
scored = [item for item in records if item["val_loss"] is not None]
if scored:
    top1 = min(scored, key=lambda item: (item["val_loss"], -item["step"]))
elif records:
    top1 = max(records, key=lambda item: item["step"])

latest = {"name": "latest_final", "path": str(stage_b / "final"), "stage": "stage_b", "step": None, "val_loss": None}
rows = []
keep_paths = set()
if top1 and Path(top1["path"]).exists():
    rows.append((top1["name"], top1["path"]))
    keep_paths.add(str(Path(top1["path"]).resolve()))
if (stage_b / "final").exists():
    if not rows or Path(latest["path"]).resolve() != Path(rows[-1][1]).resolve():
        rows.append((latest["name"], latest["path"]))

manifest.write_text("".join(f"{name}\t{path}\n" for name, path in rows), encoding="utf-8")

# After Stage B has loaded from Stage A, do not keep intermediate Stage A final.
removed = []
stage_a_final = stage_a / "final"
if os.environ.get("START_AT_STAGE_B") != "1" and stage_a_final.exists():
    shutil.rmtree(stage_a_final)
    removed.append(str(stage_a_final))

payload = {
    "policy": "sample only top1 val-loss step checkpoint plus latest/final",
    "top1": top1,
    "latest": latest if (stage_b / "final").exists() else None,
    "manifest": str(manifest),
    "manifest_rows": [{"name": name, "path": path} for name, path in rows],
    "removed_after_continuation": removed,
}
(notes_dir / f"{label}_top1_latest_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [ ! -s "${CANDIDATE_MANIFEST}" ]; then
  echo "No checkpoints found for candidate evaluation." >&2
  exit 1
fi

run_candidate_smoke() {
  local candidate_name="$1"
  local checkpoint_path="$2"
  local candidate_dir="${BRANCH_DIR}/candidate_smoke/${candidate_name}"
  local sample_dir="${candidate_dir}/sample256"
  local notes_dir="${candidate_dir}/notes"
  local log_file="${LOG_DIR}/${MODEL_LABEL}_${candidate_name}_sample256.log"
  mkdir -p "${sample_dir}" "${notes_dir}"

  local port
  next_port
  port="${NEXT_PORT}"
  run_logged "${log_file}" \
    torchrun --nproc_per_node=2 --master_port "${port}" scripts/sample_llada_crystals.py \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${checkpoint_path}" \
      --output-dir "${sample_dir}" \
      --num-samples 256 \
      --batch-size "${SAMPLE_BATCH_SIZE}" \
      --block-length 1 \
      --temperature "${TEMPERATURE}" \
      --generation-schedule "${GENERATION_SCHEDULE}" \
      --schema-logit-mask \
      --prefill-slot-tokens \
      --atom-count-grammar-mask \
      --duplicate-coordinate-mask \
      --lattice-volume-mask

  python scripts/analyze_sample_outputs.py \
    --input-jsonl "${sample_dir}/raw_generations.jsonl" \
    --failure-jsonl "${sample_dir}/failure_cases.jsonl" \
    --output-json "${notes_dir}/sample256_distribution.json" \
    --output-md "${notes_dir}/sample256_distribution.md"

  python scripts/analyze_composition_validity.py \
    --raw-generations-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_dir}/sample256_composition.json" \
    --output-md "${notes_dir}/sample256_composition.md"

  python scripts/analyze_composition_failure_modes.py \
    --raw-jsonl "${sample_dir}/raw_generations.jsonl" \
    --output-json "${notes_dir}/sample256_failure_modes.json" \
    --output-md "${notes_dir}/sample256_failure_modes.md"
}

while IFS=$'\t' read -r candidate_name checkpoint_path; do
  run_candidate_smoke "${candidate_name}" "${checkpoint_path}"
done < "${CANDIDATE_MANIFEST}"

python - <<'PY'
import json
import math
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
branch_dir = Path(os.environ["BRANCH_DIR"])
notes_dir = Path(os.environ["NOTES_DIR"])
manifest = Path(os.environ["CANDIDATE_MANIFEST"])
stage_a = Path(os.environ["STAGE_A_DIR"])
stage_b = Path(os.environ["STAGE_B_DIR"])
label = os.environ["MODEL_LABEL"]
model_path = os.environ["MODEL_PATH"]

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def nearest_val_loss(checkpoint: str) -> float | None:
    path = Path(checkpoint)
    if stage_a in path.parents or path == stage_a / "final":
        log = stage_a / "training_log.jsonl"
        target = None
    elif stage_b in path.parents or path == stage_b / "final":
        log = stage_b / "training_log.jsonl"
        target = None
        if path.name.startswith("step-"):
            try:
                target = int(path.name.split("-", 1)[1])
            except Exception:
                target = None
    else:
        return None
    losses = []
    if not log.exists():
        return None
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("event") == "eval" and "val_loss" in item:
                step = int(item.get("step") or 0)
                if target is None or step <= target:
                    losses.append((step, float(item["val_loss"])))
    return losses[-1][1] if losses else None

records = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    name, checkpoint = line.split("\t", 1)
    candidate_dir = branch_dir / "candidate_smoke" / name
    sample = read_json(candidate_dir / "sample256" / "sample_metrics.json")
    comp = read_json(candidate_dir / "notes" / "sample256_composition.json")
    failure = read_json(candidate_dir / "notes" / "sample256_failure_modes.json")
    raw = comp.get("raw_jsonl", {}) if isinstance(comp.get("raw_jsonl"), dict) else {}
    reasons = raw.get("reason_counts", {}) if isinstance(raw.get("reason_counts"), dict) else {}
    count = int(raw.get("count") or sample.get("decoded_samples") or 0)
    strict = (float(reasons.get("charge_neutral_pauling_valid", 0)) / count) if count else 0.0
    single = (float(reasons.get("single_element_shortcut", 0)) / count) if count else 0.0
    all_metal = (float(reasons.get("all_metal_shortcut", 0)) / count) if count else 0.0
    parse_rate = sample.get("parse_rate")
    if parse_rate is None:
        parse_rate = float(sample.get("parse_success", 0)) / max(1, int(sample.get("decoded_samples") or 0))
    graph_rate = sample.get("graph_acceptance_rate", sample.get("graph_rate"))
    if graph_rate is None:
        graph_rate = float(sample.get("graph_success", 0)) / max(1, int(sample.get("decoded_samples") or 0))
    comp_valid = raw.get("comp_valid_rate")
    if comp_valid is None:
        comp_valid = 0.0
    val_loss = nearest_val_loss(checkpoint)
    gate_pass = float(parse_rate) >= 0.98 and float(graph_rate) >= 0.95
    records.append({
        "name": name,
        "checkpoint_path": checkpoint,
        "sample_metrics": str(candidate_dir / "sample256" / "sample_metrics.json"),
        "composition": str(candidate_dir / "notes" / "sample256_composition.json"),
        "failure_modes": str(candidate_dir / "notes" / "sample256_failure_modes.json"),
        "parse_rate": float(parse_rate),
        "graph_acceptance_rate": float(graph_rate),
        "comp_valid": float(comp_valid),
        "strict_valid": float(strict),
        "single_element": float(single),
        "all_metal": float(all_metal),
        "pbc_duplicate": raw.get("pbc_equivalent_duplicate_fraction"),
        "val_loss": val_loss,
        "gate_pass": gate_pass,
        "headline": failure.get("headline", []),
    })

if not records:
    raise SystemExit("no candidate records")

if any(item["gate_pass"] for item in records):
    pool = [item for item in records if item["gate_pass"]]
    failed_quality = False
    best = sorted(
        pool,
        key=lambda item: (
            item["comp_valid"],
            item["strict_valid"],
            -item["single_element"],
            -(item["val_loss"] if item["val_loss"] is not None else math.inf),
            item["graph_acceptance_rate"],
        ),
        reverse=True,
    )[0]
else:
    failed_quality = True
    best = sorted(
        records,
        key=lambda item: (
            item["graph_acceptance_rate"],
            item["parse_rate"],
            item["comp_valid"],
            item["strict_valid"],
            -item["single_element"],
        ),
        reverse=True,
    )[0]

payload = {
    "model_label": label,
    "model_path": model_path,
    "failed_quality": failed_quality,
    "best": best,
    "candidates": records,
}
notes_dir.mkdir(parents=True, exist_ok=True)
(notes_dir / f"{label}_candidate_selection.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(notes_dir / f"{label}_best_checkpoint.txt").write_text(best["checkpoint_path"] + "\n", encoding="utf-8")

lines = [
    f"# Candidate Selection: {label}",
    "",
    f"- failed_quality: `{failed_quality}`",
    f"- best: `{best['name']}`",
    f"- checkpoint: `{best['checkpoint_path']}`",
    "",
    "| candidate | gate | parse | graph | comp_valid | strict | single | all_metal | val_loss |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in sorted(records, key=lambda x: (x["gate_pass"], x["comp_valid"], x["strict_valid"], -x["single_element"]), reverse=True):
    def pct(value):
        return f"{100.0 * float(value):.2f}%"
    val_loss = "n/a" if item["val_loss"] is None else f"{item['val_loss']:.4f}"
    lines.append(
        f"| {item['name']} | {item['gate_pass']} | {pct(item['parse_rate'])} | "
        f"{pct(item['graph_acceptance_rate'])} | {pct(item['comp_valid'])} | "
        f"{pct(item['strict_valid'])} | {pct(item['single_element'])} | "
        f"{pct(item['all_metal'])} | {val_loss} |"
    )
(notes_dir / f"{label}_candidate_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload["best"], ensure_ascii=False, indent=2))
PY

BEST_CHECKPOINT="$(cat "${NOTES_DIR}/${MODEL_LABEL}_best_checkpoint.txt")"
export BEST_CHECKPOINT
FAILED_QUALITY="$(python - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads((Path(os.environ["NOTES_DIR"]) / f"{os.environ['MODEL_LABEL']}_candidate_selection.json").read_text())
print("1" if payload.get("failed_quality") else "0")
PY
)"
if [ "${FAILED_QUALITY}" = "1" ]; then
  python - <<'PY'
import json
import os
from pathlib import Path
payload = {
    "model_label": os.environ["MODEL_LABEL"],
    "model_path": os.environ["MODEL_PATH"],
    "status": "stopped_after_256_failed_quality",
    "best_checkpoint": os.environ["BEST_CHECKPOINT"],
    "candidate_selection": str(Path(os.environ["NOTES_DIR"]) / f"{os.environ['MODEL_LABEL']}_candidate_selection.json"),
}
(Path(os.environ["NOTES_DIR"]) / f"{os.environ['MODEL_LABEL']}_result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  echo "All 256 candidates failed parse/graph gate; stopping before 1000 evaluation."
  exit 0
fi

SAMPLE1000_DIR="${BRANCH_DIR}/sample1000"
REFINED1000_DIR="${BRANCH_DIR}/refined1000"
SUN1000_DIR="${BRANCH_DIR}/mattergen_sun1000"
export SAMPLE1000_DIR REFINED1000_DIR SUN1000_DIR
mkdir -p "${SAMPLE1000_DIR}" "${REFINED1000_DIR}" "${SUN1000_DIR}"

next_port
port="${NEXT_PORT}"
run_logged "${LOG_DIR}/${MODEL_LABEL}_sample1000.log" \
  torchrun --nproc_per_node=2 --master_port "${port}" scripts/sample_llada_crystals.py \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${BEST_CHECKPOINT}" \
    --output-dir "${SAMPLE1000_DIR}" \
    --target-graph-success "${TARGET_GRAPH_SUCCESS}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --num-samples "${MAX_ATTEMPTS}" \
    --batch-size "${SAMPLE_BATCH_SIZE}" \
    --block-length 1 \
    --temperature "${TEMPERATURE}" \
    --generation-schedule "${GENERATION_SCHEDULE}" \
    --schema-logit-mask \
    --prefill-slot-tokens \
    --atom-count-grammar-mask \
    --duplicate-coordinate-mask \
    --lattice-volume-mask

python scripts/analyze_sample_outputs.py \
  --input-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --failure-jsonl "${SAMPLE1000_DIR}/failure_cases.jsonl" \
  --output-json "${NOTES_DIR}/${MODEL_LABEL}_sample1000_distribution.json" \
  --output-md "${NOTES_DIR}/${MODEL_LABEL}_sample1000_distribution.md"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/${MODEL_LABEL}_sample1000_composition_raw.json" \
  --output-md "${NOTES_DIR}/${MODEL_LABEL}_sample1000_composition_raw.md"

python scripts/analyze_composition_failure_modes.py \
  --raw-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --output-json "${NOTES_DIR}/${MODEL_LABEL}_sample1000_failure_modes_raw.json" \
  --output-md "${NOTES_DIR}/${MODEL_LABEL}_sample1000_failure_modes_raw.md"

TARGET_REACHED="$(python - <<'PY'
import json
import os
from pathlib import Path
metrics = json.loads((Path(os.environ["SAMPLE1000_DIR"]) / "sample_metrics.json").read_text())
print("1" if metrics.get("target_reached") else "0")
PY
)"
if [ "${TARGET_REACHED}" != "1" ]; then
  python - <<'PY'
import json
import os
from pathlib import Path
payload = {
    "model_label": os.environ["MODEL_LABEL"],
    "model_path": os.environ["MODEL_PATH"],
    "status": "stopped_after_1000_target_not_reached",
    "best_checkpoint": os.environ["BEST_CHECKPOINT"],
    "sample_metrics": str(Path(os.environ["SAMPLE1000_DIR"]) / "sample_metrics.json"),
}
(Path(os.environ["NOTES_DIR"]) / f"{os.environ['MODEL_LABEL']}_result_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  echo "1000 graph-valid target was not reached; stopping before refinement."
  exit 0
fi

next_port
port="${NEXT_PORT}"
run_logged "${LOG_DIR}/${MODEL_LABEL}_refine1000.log" \
  torchrun --nproc_per_node=2 --master_port "${port}" scripts/refine_dlm_with_crysllmgen.py \
    --proposal-graphs "${SAMPLE1000_DIR}/proposal_graphs.pt" \
    --checkpoint "${CRYSLLMGEN_CHECKPOINT}" \
    --output-dir "${REFINED1000_DIR}" \
    --batch-size "${REFINE_BATCH_SIZE}" \
    --diff-steps "${DIFF_STEPS}" \
    --max-proposals "${TARGET_GRAPH_SUCCESS}"

python scripts/run_crysllmgen_metrics.py \
  --root-path "${REFINED1000_DIR}" \
  --output-json "${NOTES_DIR}/${MODEL_LABEL}_crysllmgen_metrics1000.json"

python scripts/analyze_composition_validity.py \
  --raw-generations-jsonl "${SAMPLE1000_DIR}/raw_generations.jsonl" \
  --refined-pt "${REFINED1000_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt" \
  --refined-world-size "${REFINED_WORLD_SIZE}" \
  --output-json "${NOTES_DIR}/${MODEL_LABEL}_composition1000.json" \
  --output-md "${NOTES_DIR}/${MODEL_LABEL}_composition1000.md"

"${MATTERGEN_PYTHON}" scripts/convert_crysllmgen_pt_to_extxyz.py \
  --input-pt "${REFINED1000_DIR}/dlm_refined_mp_${TARGET_GRAPH_SUCCESS}.pt" \
  --output-extxyz "${SUN1000_DIR}/generated.extxyz"

sun_args=(
  --structures-path "${SUN1000_DIR}/generated.extxyz"
  --reference-dataset "${REFERENCE_DATASET}"
  --save-as "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_metrics.json"
  --save-detailed-as "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_detailed_metrics.json"
  --structures-output-path "${SUN1000_DIR}/relaxed.extxyz"
  --summary-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_summary.json"
  --relax-failures-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_relax_failures.json"
  --unsupported-failures-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_unsupported_failures.json"
  --metric-errors-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_metric_errors.json"
  --relax-max-steps "${RELAX_MAX_STEPS:-500}"
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH:-512}"
  --device cuda
  --structure-matcher disordered
)
if [ -f "${MATTERSIM_CHECKPOINT}" ]; then
  sun_args+=(--potential-load-path "${MATTERSIM_CHECKPOINT}")
fi
"${MATTERGEN_PYTHON}" scripts/run_mattergen_sun_eval.py "${sun_args[@]}"

if [ -f scripts/analyze_mattergen_sun_detailed.py ]; then
  python scripts/analyze_mattergen_sun_detailed.py \
    --summary-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_summary.json" \
    --detailed-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_detailed_metrics.json" \
    --label "${MODEL_LABEL}" \
    --output-json "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_threshold_analysis.json" \
    --output-md "${NOTES_DIR}/${MODEL_LABEL}_mattergen_sun1000_threshold_analysis.md"
fi

python - <<'PY'
import json
import os
from pathlib import Path
from typing import Any, Mapping

run_id = os.environ["RUN_ID"]
label = os.environ["MODEL_LABEL"]
model_path = os.environ["MODEL_PATH"]
generation_schedule = os.environ["GENERATION_SCHEDULE"]
notes = Path(os.environ["NOTES_DIR"])
reports = Path(os.environ["REPORT_DIR"])
sample1000_dir = Path(os.environ["SAMPLE1000_DIR"])

def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def get_metric(payload: Mapping[str, Any], key: str) -> Any:
    metrics = payload.get("metrics", payload)
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
        if isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value
    return None

def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"

def raw(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"

def comp_scope(comp: Mapping[str, Any], key: str) -> dict[str, Any]:
    scope = comp.get(key, {})
    if not isinstance(scope, Mapping):
        return {}
    reasons = scope.get("reason_counts", {}) if isinstance(scope.get("reason_counts"), Mapping) else {}
    total = float(scope.get("count") or 0)
    def reason_rate(reason: str):
        return None if total <= 0 else float(reasons.get(reason, 0)) / total
    return {
        "count": scope.get("count"),
        "comp_valid": scope.get("comp_valid_rate"),
        "strict": reason_rate("charge_neutral_pauling_valid"),
        "single": reason_rate("single_element_shortcut"),
        "all_metal": reason_rate("all_metal_shortcut"),
        "pbc_duplicate": scope.get("pbc_equivalent_duplicate_fraction"),
        "top_reasons": dict(list(reasons.items())[:10]),
    }

selection = read_json(notes / f"{label}_candidate_selection.json")
sample = read_json(sample1000_dir / "sample_metrics.json")
comp = read_json(notes / f"{label}_composition1000.json")
crys = read_json(notes / f"{label}_crysllmgen_metrics1000.json")
sun = read_json(notes / f"{label}_mattergen_sun1000_summary.json")
failure = read_json(notes / f"{label}_sample1000_failure_modes_raw.json")
raw_comp = comp_scope(comp, "raw_jsonl")
refined_comp = comp_scope(comp, "refined_pt")
thresholds = sun.get("sun_thresholds") or {}
rates = thresholds.get("rates") or {}
rates_success = thresholds.get("rates_among_successful") or {}

summary = {
    "model_label": label,
    "model_path": model_path,
    "status": "completed",
    "best_checkpoint": selection.get("best", {}).get("checkpoint_path"),
    "candidate_selection": str(notes / f"{label}_candidate_selection.json"),
    "sample_metrics": str(sample1000_dir / "sample_metrics.json"),
    "composition": str(notes / f"{label}_composition1000.json"),
    "crysllmgen_metrics": str(notes / f"{label}_crysllmgen_metrics1000.json"),
    "mattergen_sun_summary": str(notes / f"{label}_mattergen_sun1000_summary.json"),
    "strict_sun": rates.get("strict_sun"),
    "meta_sun": rates.get("meta_sun"),
}
(notes / f"{label}_result_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

crys_keys = ["comp_valid", "struct_valid", "valid", "wdist_density", "wdist_num_elems", "cov_recall", "cov_precision"]
sun_keys = [
    "frac_novel_unique_stable_structures",
    "frac_stable_structures",
    "frac_novel_structures",
    "frac_unique_structures",
    "frac_novel_unique_structures",
    "frac_successful_jobs",
    "avg_energy_above_hull_per_atom",
]
lines = [
    f"# MP-20 Raw SFT Baseline 1000 Evaluation: {label}",
    "",
    "## Configuration",
    "",
    f"- run_id: `{run_id}`",
    f"- model_path: `{model_path}`",
    f"- best_checkpoint: `{summary['best_checkpoint']}`",
    f"- SFT data: `data/dlm_sft/mp_20`",
    f"- sampling: `temperature=0.7`, `block_length=1`, `{generation_schedule}`, schema/slot/count/duplicate/lattice masks enabled",
    "",
    "## 256 Checkpoint Selection",
    "",
    f"- selection file: `{notes / f'{label}_candidate_selection.md'}`",
    f"- failed_quality: `{selection.get('failed_quality')}`",
    "",
    "## 1000 Sampling",
    "",
    "| metric | value |",
    "| --- | ---: |",
    f"| decoded_samples | {sample.get('decoded_samples', 'n/a')} |",
    f"| graph_success | {sample.get('graph_success', 'n/a')} |",
    f"| target_reached | {sample.get('target_reached', 'n/a')} |",
    f"| parse_rate | {pct(sample.get('parse_rate'))} |",
    f"| graph_acceptance_rate | {pct(sample.get('graph_acceptance_rate'))} |",
    "",
    "## Composition",
    "",
    "| scope | count | comp_valid | strict | single | all_metal | PBC duplicate |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    f"| raw | {raw_comp.get('count', 'n/a')} | {pct(raw_comp.get('comp_valid'))} | {pct(raw_comp.get('strict'))} | {pct(raw_comp.get('single'))} | {pct(raw_comp.get('all_metal'))} | {pct(raw_comp.get('pbc_duplicate'))} |",
    f"| refined | {refined_comp.get('count', 'n/a')} | {pct(refined_comp.get('comp_valid'))} | {pct(refined_comp.get('strict'))} | {pct(refined_comp.get('single'))} | {pct(refined_comp.get('all_metal'))} | {pct(refined_comp.get('pbc_duplicate'))} |",
    "",
    "Top refined reasons:",
    "",
    "```json",
    json.dumps(refined_comp.get("top_reasons", {}), ensure_ascii=False, indent=2),
    "```",
    "",
    "Raw comp_valid bottleneck:",
    "",
    "```json",
    json.dumps(failure.get("headline", []), ensure_ascii=False, indent=2),
    "```",
    "",
    "## CrysLLMGen Metrics",
    "",
    "| metric | value |",
    "| --- | ---: |",
]
for key in crys_keys:
    value = get_metric(crys, key)
    if key.startswith("wdist"):
        lines.append(f"| {key} | {raw(value)} |")
    else:
        lines.append(f"| {key} | {raw(value)} |")
lines.extend(["", "## MatterGen S.U.N", "", "| metric | value |", "| --- | ---: |"])
for key in sun_keys:
    value = get_metric(sun, key)
    lines.append(f"| {key} | {raw(value)} |")
lines.extend([
    f"| strict_stable submitted | {pct(rates.get('strict_stable'))} |",
    f"| meta_stable submitted | {pct(rates.get('meta_stable'))} |",
    f"| strict_sun submitted | {pct(rates.get('strict_sun'))} |",
    f"| meta_sun submitted | {pct(rates.get('meta_sun'))} |",
    f"| strict_sun successful | {pct(rates_success.get('strict_sun'))} |",
    f"| meta_sun successful | {pct(rates_success.get('meta_sun'))} |",
    "",
    "## Files",
    "",
    f"- branch summary: `{notes / f'{label}_result_summary.json'}`",
    f"- CrysLLMGen metrics: `{notes / f'{label}_crysllmgen_metrics1000.json'}`",
    f"- MatterGen summary: `{notes / f'{label}_mattergen_sun1000_summary.json'}`",
    f"- strict/meta threshold analysis: `{notes / f'{label}_mattergen_sun1000_threshold_analysis.json'}`",
])
reports.mkdir(parents=True, exist_ok=True)
report_path = reports / f"{run_id}_{label}_raw_sft1000_crysllmgen_sun.md"
report_text = "\n".join(lines) + "\n"
report_path.write_text(report_text, encoding="utf-8")
(notes / f"{label}_report.md").write_text(report_text, encoding="utf-8")
print(f"REPORT={report_path}")
PY

echo "Completed raw SFT baseline branch ${MODEL_LABEL}."

#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-native-s17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=pcdlm-native-s17-%j.out
#SBATCH --error=pcdlm-native-s17-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly TRAIN_RUN="${POTENTIAL_CLOSURE_TRAIN_RUN:?}"
readonly PLAN_DIR="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_native_s17_${SLURM_JOB_ID}"
readonly STREAM=17
readonly DLM_SEED=91117

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

[[ -f "${TRAIN_RUN}/_SUCCESS" ]]
[[ -f "${PLAN_DIR}/_SUCCESS" ]]
[[ "$(wc -l <"${PLAN_DIR}/ledger.jsonl")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

policy_path() {
  local mode="$1"
  "${PYTHON}" -c 'import json,pathlib,sys; x=json.load(open(pathlib.Path(sys.argv[1])/sys.argv[2]/"TRAIN_FINAL.json")); print(x["checkpoint"]["policy_adapter_path"])' \
    "${TRAIN_RUN}" "${mode}"
}

run_cell() {
  local mode="$1" gpu="$2" policy cell
  policy="$(policy_path "${mode}")"
  cell="${RUN}/${mode}/stream17"
  mkdir -p "${cell}"
  [[ -f "${policy}/adapter_model.safetensors" ]]
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" --checkpoint-path "${policy}" \
    --prompt-jsonl "${PLAN_DIR}/plans_for_dlm.jsonl" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" \
    --num-samples 256 --batch-size 8 --temperature 0.7 --seed "${DLM_SEED}" \
    --preserve-prompt-sample-idx --no-repeat-prompt-records \
    --generation-schedule spad --canonicalize-periodic-alias \
    --pbc-min-distance-mask --spad-cell-closure --spad-backfill \
    --spad-max-anchor-revisions 2 \
    >"${RUN}/logs/${mode}_body.out" 2>"${RUN}/logs/${mode}_body.err"
  [[ -f "${cell}/body/_SUCCESS" ]]

  "${PYTHON}" "${SOURCE}/scripts/pad_spad_body_to_planner_denominator.py" \
    --body-rows "${cell}/body/raw_generations.jsonl" \
    --planner-ledger "${PLAN_DIR}/ledger.jsonl" \
    --output-dir "${cell}/body_fixed256" --denominator 256
  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body_fixed256" --output-dir "${cell}/raw_generation" \
    --seed 17 --dlm-seed "${DLM_SEED}" --denominator 256 \
    --source-arm full_axis --arm-label candidate
  touch "${cell}/_SUCCESS"
}

run_cell closure_control "${allocated[0]}" & control_pid="$!"
run_cell potential_closed "${allocated[1]}" & potential_pid="$!"
status=0
wait "${control_pid}" || status="$?"
wait "${potential_pid}" || status="$?"
(( status == 0 ))

"${PYTHON}" - "${RUN}" "${TRAIN_RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
train = str(pathlib.Path(sys.argv[2]).resolve())
cells = []
for mode in ("closure_control", "potential_closed"):
    cell = root / mode / "stream17"
    metrics = json.load(open(cell / "body/sample_metrics.json"))
    rows = [json.loads(line) for line in open(cell / "body/raw_generations.jsonl") if line.strip()]
    cell_changed = sum(
        bool(row.get("spad_cell_revision_log"))
        and int(row["spad_cell_revision_log"].get("changed_components", 0)) > 0
        for row in rows
    )
    anchor_changed = sum(
        sum(int(item.get("changed_components", 0)) > 0 for item in row.get("spad_revision_log", []))
        for row in rows
    )
    cells.append({
        "mode": mode,
        "requested": 256,
        "parsed": int(metrics["parse_success"]),
        "plan_match": int(metrics["plan_match_success"]),
        "graphs": int(metrics["graph_success"]),
        "cell_transactions_changed": int(cell_changed),
        "anchor_transactions_changed": int(anchor_changed),
    })
report = {
    "schema": "potential_closure_native_stream17_generation_v1",
    "training_run": train,
    "stream": 17,
    "dlm_seed": 91117,
    "cells": cells,
    "schedule": "SPAD body -> one cell closure -> first two Llama anchors in reverse order",
    "one_plan_one_trajectory": True,
    "model494_or_inference_critic": False,
    "selection_retry_replacement": False,
}
(root / "GENERATION_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=spad-e-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=spad-e-train-%j.out
#SBATCH --error=spad-e-train-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly TEACHER="${SPAD_ENERGY_TEACHER_RUN:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly VOCAB_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_energy_train_${SLURM_JOB_ID}"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${TEACHER}/_SUCCESS" && -f "${TEACHER}/labels/_SUCCESS" ]]
[[ "$(wc -l <"${TEACHER}/labels/groups.jsonl")" -eq 2048 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

run_cell() {
  local mode="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/train_spad_energy.py" \
    --model-path "${MODEL}" --checkpoint-path "${BS_POLICY}" \
    --groups "${TEACHER}/labels/groups.jsonl" --vocab-data-dir "${VOCAB_DATA}" \
    --output-dir "${RUN}/${mode}" --mode "${mode}" --seed 98017 \
    >"${RUN}/logs/${mode}.out" 2>"${RUN}/logs/${mode}.err"
}

run_cell control "${allocated[0]}" & control_pid="$!"
run_cell energy "${allocated[1]}" & energy_pid="$!"
status=0
wait "${control_pid}" || status="$?"
wait "${energy_pid}" || status="$?"
(( status == 0 ))

"${PYTHON}" - "${RUN}" "${TEACHER}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
teacher = str(pathlib.Path(sys.argv[2]).resolve())
cells = []
for mode in ("control", "energy"):
    final = json.load(open(root / mode / "TRAIN_FINAL.json"))
    assert final["status"] == "success" and final["updates"] == 348
    assert final["mode"] == mode and final["seed"] == 98017
    assert final["groups_total"] == 2048
    cells.append({
        "mode": mode,
        "updates": final["updates"],
        "groups_trainable": final["groups_trainable"],
        "groups_untrainable": final["groups_untrainable"],
        "checkpoint": final["checkpoint"]["policy_adapter_path"],
        "elapsed_sec": final["elapsed_sec"],
    })
report = {
    "schema": "spad_energy_dual_cell_train_v1",
    "teacher": teacher,
    "cells": cells,
    "same_seed_data_updates": True,
    "validity_before_energy": True,
    "selection_or_grid": False,
    "inference_time_critic": False,
}
(root / "TRAIN_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"

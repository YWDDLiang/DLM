#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=pcdlm-train-%j.out
#SBATCH --error=pcdlm-train-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly LABEL_DIR="${POTENTIAL_CLOSURE_LABEL_DIR:?}"
readonly PROBE_RUN="${POTENTIAL_CLOSURE_PROBE_RUN:?}"
readonly PROBE_REPORT="${PROBE_RUN}/probe/PROBE_FINAL.json"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly SFT_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_train_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${PROBE_RUN}/_SUCCESS" ]]
[[ -f "${PROBE_RUN}/probe/_SUCCESS" ]]
[[ -f "${PROBE_REPORT}" ]]
[[ -f "${LABEL_DIR}/_SUCCESS" ]]
[[ -f "${LABEL_DIR}/labelled_groups.jsonl" ]]
[[ -f "${LABEL_DIR}/manifest.json" ]]
[[ -f "${SFT_DATA}/_SUCCESS" ]]
[[ -f "${BS_POLICY}/adapter_model.safetensors" ]]
"${PYTHON}" - "${PROBE_REPORT}" "${LABEL_DIR}/manifest.json" <<'PY'
import json, pathlib, sys
probe = json.load(open(pathlib.Path(sys.argv[1])))
actions = json.load(open(pathlib.Path(sys.argv[2])))
assert probe["schema"] == "potential_closure_gradient_probe_v1"
assert probe["passed"] is True and probe["optimizer_updates"] == 0
assert actions["schema"] == "potential_closure_label_manifest_v1"
assert actions["formal_action_pool_gate"] is True
assert int(actions["groups"]) == 2048
PY
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

run_cell() {
  local mode="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
    "${SOURCE}/src/scripts/train_potential_closure.py" \
    --model-path "${MODEL}" --checkpoint-path "${BS_POLICY}" \
    --labelled-groups "${LABEL_DIR}/labelled_groups.jsonl" \
    --action-manifest "${LABEL_DIR}/manifest.json" \
    --sft-data-dir "${SFT_DATA}" --output-dir "${RUN}/${mode}" \
    --mode "${mode}" --seed 99017 --probe-report "${PROBE_REPORT}" \
    >"${RUN}/logs/${mode}.out" 2>"${RUN}/logs/${mode}.err"
}

run_cell closure_control "${allocated[0]}" & control_pid="$!"
run_cell potential_closed "${allocated[1]}" & potential_pid="$!"
status=0
wait "${control_pid}" || status="$?"
wait "${potential_pid}" || status="$?"
(( status == 0 ))

"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cells = []
for mode in ("closure_control", "potential_closed"):
    cell = root / mode
    assert (cell / "_SUCCESS").is_file()
    final = json.load(open(cell / "TRAIN_FINAL.json"))
    assert final["schema"] == "potential_closure_train_v1"
    assert final["status"] == "success" and final["mode"] == mode
    assert final["seed"] == 99017 and final["optimizer_updates"] == 2048
    assert final["gradient_accumulation"] == 6
    assert final["objective_counts"] == {"cell": 512, "clean_ce": 1024, "site": 512}
    checkpoints = [path.name for path in (cell / "checkpoints").iterdir() if path.is_dir()]
    assert checkpoints == ["step-2048"]
    assert final["checkpoint"]["reference_adapter_removed"] is True
    cells.append(final)
(root / "TRAIN_FINAL.json").write_text(json.dumps({
    "schema": "potential_closure_dual_mode_train_v1",
    "status": "success",
    "same_bs_initialization_seed_optimizer_updates": True,
    "modes": [cell["mode"] for cell in cells],
    "cells": cells,
}, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

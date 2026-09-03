#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-pot-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=pcdlm-pot-recover-%j.out
#SBATCH --error=pcdlm-pot-recover-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly LABEL_DIR="${POTENTIAL_CLOSURE_LABEL_DIR:?}"
readonly PROBE_RUN="${POTENTIAL_CLOSURE_PROBE_RUN:?}"
readonly FAILED_PARENT_RUN="${POTENTIAL_CLOSURE_FAILED_PARENT_RUN:?}"
readonly PROBE_REPORT="${PROBE_RUN}/probe/PROBE_FINAL.json"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly SFT_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closed_precision_recovery_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

[[ -f "${FAILED_PARENT_RUN}/_FAILED" ]]
[[ -f "${FAILED_PARENT_RUN}/potential_closed/_FAILED.json" ]]
[[ -f "${PROBE_RUN}/_SUCCESS" ]]
[[ -f "${PROBE_REPORT}" ]]
[[ -f "${LABEL_DIR}/_SUCCESS" ]]
[[ -f "${LABEL_DIR}/labelled_groups.jsonl" ]]
[[ -f "${LABEL_DIR}/manifest.json" ]]
[[ -f "${SFT_DATA}/_SUCCESS" ]]
[[ -f "${BS_POLICY}/adapter_model.safetensors" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/src/scripts/train_potential_closure.py" \
  --model-path "${MODEL}" --checkpoint-path "${BS_POLICY}" \
  --labelled-groups "${LABEL_DIR}/labelled_groups.jsonl" \
  --action-manifest "${LABEL_DIR}/manifest.json" \
  --sft-data-dir "${SFT_DATA}" --output-dir "${RUN}/potential_closed" \
  --mode potential_closed --seed 99017 --probe-report "${PROBE_REPORT}" \
  >"${RUN}/logs/potential_closed.out" 2>"${RUN}/logs/potential_closed.err"

"${PYTHON}" - "${RUN}" "${FAILED_PARENT_RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
parent = pathlib.Path(sys.argv[2])
cell = root / "potential_closed"
assert (cell / "_SUCCESS").is_file()
final = json.load(open(cell / "TRAIN_FINAL.json"))
assert final["schema"] == "potential_closure_train_v1"
assert final["status"] == "success" and final["mode"] == "potential_closed"
assert final["seed"] == 99017 and final["optimizer_updates"] == 2048
assert final["gradient_accumulation"] == 6
assert final["objective_counts"] == {"cell": 512, "clean_ce": 1024, "site": 512}
checkpoints = [path.name for path in (cell / "checkpoints").iterdir() if path.is_dir()]
assert checkpoints == ["step-2048"]
assert final["checkpoint"]["reference_adapter_removed"] is True
(root / "RECOVERY_FINAL.json").write_text(json.dumps({
    "schema": "potential_closed_precision_recovery_v1",
    "status": "success",
    "failed_parent_run": str(parent),
    "scientific_configuration_unchanged": True,
    "numeric_fix": "preserve the float64-normalized posterior in the K<=4 loss",
    "cell": final,
}, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

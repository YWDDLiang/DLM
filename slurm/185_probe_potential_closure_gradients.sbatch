#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=pcdlm-probe-%j.out
#SBATCH --error=pcdlm-probe-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly LABEL_DIR="${POTENTIAL_CLOSURE_LABEL_DIR:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly SFT_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_gradient_probe_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${LABEL_DIR}/_SUCCESS" ]]
[[ -f "${LABEL_DIR}/labelled_groups.jsonl" ]]
[[ -f "${LABEL_DIR}/manifest.json" ]]
[[ -f "${SFT_DATA}/_SUCCESS" ]]
[[ -f "${BS_POLICY}/adapter_model.safetensors" ]]
"${PYTHON}" - "${LABEL_DIR}/manifest.json" <<'PY'
import json, pathlib, sys
value = json.load(open(pathlib.Path(sys.argv[1])))
assert value["schema"] == "potential_closure_label_manifest_v1"
assert value["formal_action_pool_gate"] is True
assert int(value["groups"]) == 2048
PY
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/src/scripts/train_potential_closure.py" \
  --model-path "${MODEL}" --checkpoint-path "${BS_POLICY}" \
  --labelled-groups "${LABEL_DIR}/labelled_groups.jsonl" \
  --action-manifest "${LABEL_DIR}/manifest.json" \
  --sft-data-dir "${SFT_DATA}" --output-dir "${RUN}/probe" \
  --mode potential_closed --seed 99017 --probe-only \
  >"${RUN}/logs/probe.out" 2>"${RUN}/logs/probe.err"

[[ -f "${RUN}/probe/_SUCCESS" ]]
"${PYTHON}" - "${RUN}/probe/PROBE_FINAL.json" <<'PY'
import json, pathlib, sys
value = json.load(open(pathlib.Path(sys.argv[1])))
assert value["schema"] == "potential_closure_gradient_probe_v1"
assert value["passed"] is True
assert value["optimizer_updates"] == 0
assert value["automatic_weight_lr_kl_or_epoch_changes"] is False
PY
printf '%s\n' "${RUN}/probe/PROBE_FINAL.json" >"${RUN}/PROBE_REPORT"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

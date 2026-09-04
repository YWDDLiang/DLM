#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=spad-basin-raw-%j.out
#SBATCH --error=spad-basin-raw-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly GENERATION_RUN="${SPAD_BASIN_CLOSURE_GENERATION_RUN:?export completed native generation run}"
readonly BS_GENERATION="${ROOT}/runs/spad_prospective_generation_s17_39537/BS/stream17/raw_generation/generation.jsonl"
readonly CLOSURE_GENERATION="${GENERATION_RUN}/closure_ce/stream17/raw_generation/generation.jsonl"
readonly BS_BODY_CONFIG="${ROOT}/runs/spad_prospective_generation_s17_39537/BS/stream17/body/run_config.json"
readonly BS_RUN_REPORT="${ROOT}/runs/spad_prospective_generation_s17_39537/GENERATION_FINAL.json"
readonly CLOSURE_BODY_CONFIG="${GENERATION_RUN}/closure_ce/stream17/body/run_config.json"
readonly CLOSURE_RUN_REPORT="${GENERATION_RUN}/GENERATION_FINAL.json"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_native_raw_screen_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${GENERATION_RUN}/_SUCCESS" "${BS_GENERATION}" \
  "${CLOSURE_GENERATION}" "${BS_BODY_CONFIG}" "${BS_RUN_REPORT}" \
  "${CLOSURE_BODY_CONFIG}" "${CLOSURE_RUN_REPORT}" \
  "${SOURCE}/scripts/evaluate_spad_basin_closure_raw.py"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${BS_GENERATION}")" -eq 256 ]]
[[ "$(wc -l <"${CLOSURE_GENERATION}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

"${PYTHON}" - "${BS_BODY_CONFIG}" "${CLOSURE_BODY_CONFIG}" \
  "${BS_RUN_REPORT}" "${CLOSURE_RUN_REPORT}" <<'PY'
import json
import pathlib
import sys

bs_config = json.load(open(sys.argv[1], encoding="utf-8"))
closure_config = json.load(open(sys.argv[2], encoding="utf-8"))
bs_report = json.load(open(sys.argv[3], encoding="utf-8"))
closure_report = json.load(open(sys.argv[4], encoding="utf-8"))
for config in (bs_config, closure_config):
    assert config["world_size"] == 1
    assert config["batch_size"] == 8
    assert config["seed"] == 91117
    assert config["rank_seed_rule"] == "seed + rank"
    assert config["num_samples"] == 256
assert pathlib.Path(bs_config["prompt_jsonl"]).resolve(strict=True) == pathlib.Path(
    closure_config["prompt_jsonl"]
).resolve(strict=True)
assert bs_report["schema"] == "spad_prospective_generation_v1"
assert bs_report["stream"] == closure_report["stream"] == 17
assert pathlib.Path(bs_report["frozen_plan_dir"]).resolve(strict=True) == pathlib.Path(
    closure_report["plan_dir"]
).resolve(strict=True)
assert closure_report["schema"] == "spad_basin_closure_native_stream17_generation_v1"
assert closure_report["world_size"] == 1
assert closure_report["batch_size"] == 8
assert closure_report["rank_seed_rule"] == "seed + rank"
assert closure_report["one_plan_one_trajectory"] is True
assert closure_report["selection_retry_replacement"] is False
PY

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/evaluate_spad_basin_closure_raw.py" \
  --bs-generation "${BS_GENERATION}" \
  --closure-generation "${CLOSURE_GENERATION}" \
  --output-dir "${RUN}/screen" --denominator 256 \
  --device cuda:0 --batch-size 16 --bootstrap-seed 20260904 \
  >"${RUN}/logs/screen.out" 2>"${RUN}/logs/screen.err"

[[ -f "${RUN}/screen/RAW_SCREEN_FINAL.json" ]]
[[ -f "${RUN}/screen/_SUCCESS" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

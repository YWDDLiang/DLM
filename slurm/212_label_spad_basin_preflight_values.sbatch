#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-value
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=spad-basin-value-%j.out
#SBATCH --error=spad-basin-value-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly ACTIONS="${SPAD_BASIN_PREFLIGHT_ACTION_GROUPS:?export the frozen actions JSONL}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly LABELER="${SOURCE}/scripts/label_spad_basin_preflight_values.py"
readonly RUN="${SPAD_BASIN_PREFLIGHT_VALUE_RUN:-${ROOT}/runs/spad_basin_preflight_values_${SLURM_JOB_ID}}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
pid0=""
pid1=""
on_error() {
  local rc="$1" line="$2"
  set +e
  [[ -z "${pid0}" ]] || kill "${pid0}" 2>/dev/null || true
  [[ -z "${pid1}" ]] || kill "${pid1}" 2>/dev/null || true
  printf 'exit_code\t%s\nline\t%s\n' "${rc}" "${line}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO"' ERR

[[ -f "${ACTIONS}" ]]
[[ -f "${LABELER}" ]]
[[ "$(wc -l <"${ACTIONS}")" -eq 128 ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?two allocated GPUs are required}"
[[ "${#gpu[@]}" -eq 2 ]]

env CUDA_VISIBLE_DEVICES="${gpu[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${LABELER}" --candidate-groups "${ACTIONS}" \
  --output-dir "${RUN}" --device cuda --batch-size 16 --fmax 0.1 \
  --shard-rank 0 --shard-count 2 \
  >"${RUN}/logs/rank0.out" 2>"${RUN}/logs/rank0.err" &
pid0=$!

env CUDA_VISIBLE_DEVICES="${gpu[1]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${LABELER}" --candidate-groups "${ACTIONS}" \
  --output-dir "${RUN}" --device cuda --batch-size 16 --fmax 0.1 \
  --shard-rank 1 --shard-count 2 \
  >"${RUN}/logs/rank1.out" 2>"${RUN}/logs/rank1.err" &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
pid0=""
pid1=""
[[ "${status}" -eq 0 ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${LABELER}" --output-dir "${RUN}" \
  --shard-count 2 --merge-shards \
  >"${RUN}/logs/merge.out" 2>"${RUN}/logs/merge.err"

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
report = json.loads((run / "PRELIGHT_VALUE_FINAL.json").read_text())
assert report["schema"] == "spad_basin_preflight_value_final_v1"
assert report["groups"] == 128
assert report["scientific_contract"]["one_trajectory_per_terminal_legal_candidate"]
assert report["scientific_contract"]["horizons"] == [3, 5, 10, 20]
assert report["scientific_contract"]["invalid_candidates_retained"]
assert sum(1 for line in open(run / "labelled_groups.jsonl") if line.strip()) == 128
assert (run / "_SUCCESS").is_file()
PY

trap - ERR
printf '%s\n' "${RUN}"

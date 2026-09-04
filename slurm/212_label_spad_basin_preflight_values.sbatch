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
readonly EXPECTED_GROUPS="${SPAD_EXPECTED_GROUPS:-128}"
readonly VALUE_WORLD_SIZE="${SPAD_VALUE_WORLD_SIZE:-2}"
readonly K10_ONLY="${SPAD_K10_ONLY:-false}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
declare -a child_pids=()
on_error() {
  local rc="$1" line="$2"
  set +e
  for pid in "${child_pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  printf 'exit_code\t%s\nline\t%s\n' "${rc}" "${line}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO"' ERR

[[ -f "${ACTIONS}" ]]
[[ -f "${LABELER}" ]]
[[ "$(wc -l <"${ACTIONS}")" -eq "${EXPECTED_GROUPS}" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?allocated GPUs are required}"
[[ "${#gpu[@]}" -eq "${VALUE_WORLD_SIZE}" ]]
[[ "${K10_ONLY}" == true || "${K10_ONLY}" == false ]]
declare -a horizon_args=()
[[ "${K10_ONLY}" == false ]] || horizon_args=(--k10-only)

for ((rank=0; rank<VALUE_WORLD_SIZE; rank++)); do
  env CUDA_VISIBLE_DEVICES="${gpu[${rank}]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${LABELER}" --candidate-groups "${ACTIONS}" \
    --output-dir "${RUN}" --device cuda --batch-size 16 --fmax 0.1 \
    --shard-rank "${rank}" --shard-count "${VALUE_WORLD_SIZE}" \
    --expected-groups "${EXPECTED_GROUPS}" \
    ${horizon_args[@]+"${horizon_args[@]}"} \
    >"${RUN}/logs/rank${rank}.out" 2>"${RUN}/logs/rank${rank}.err" &
  child_pids+=("$!")
done

status=0
for pid in "${child_pids[@]}"; do wait "${pid}" || status=1; done
child_pids=()
[[ "${status}" -eq 0 ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${LABELER}" --output-dir "${RUN}" \
  --shard-count "${VALUE_WORLD_SIZE}" --merge-shards \
  --expected-groups "${EXPECTED_GROUPS}" \
  ${horizon_args[@]+"${horizon_args[@]}"} \
  >"${RUN}/logs/merge.out" 2>"${RUN}/logs/merge.err"

"${PYTHON}" - "${RUN}" "${EXPECTED_GROUPS}" "${K10_ONLY}" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); k10_only=sys.argv[3]=='true'
report = json.loads((run / "PRELIGHT_VALUE_FINAL.json").read_text())
assert report["schema"] == "spad_basin_preflight_value_final_v1"
assert report["groups"] == expected
assert report["scientific_contract"]["one_trajectory_per_terminal_legal_candidate"]
assert report["scientific_contract"]["horizons"] == ([10] if k10_only else [3, 5, 10, 20])
assert report["scientific_contract"]["invalid_candidates_retained"]
assert sum(1 for line in open(run / "labelled_groups.jsonl") if line.strip()) == expected
assert (run / "_SUCCESS").is_file()
PY

trap - ERR
printf '%s\n' "${RUN}"

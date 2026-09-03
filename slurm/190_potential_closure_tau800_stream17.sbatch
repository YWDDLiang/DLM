#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-tau800-s17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=320G
#SBATCH --time=03:00:00
#SBATCH --output=pcdlm-tau800-s17-%j.out
#SBATCH --error=pcdlm-tau800-s17-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly NATIVE_RUN="${POTENTIAL_CLOSURE_NATIVE_RUN:?}"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_tau800_s17_${SLURM_JOB_ID}"
readonly STREAM=17
readonly DLM_SEED=91117
readonly REFINER_SEED=101117

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${NATIVE_RUN}/_SUCCESS" ]]
[[ -f "${REFINER}" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 4 ]]

run_refine() {
  local mode="$1" gpu0="$2" gpu1="$3"
  local body="${NATIVE_RUN}/${mode}/stream17/body"
  local output="${RUN}/${mode}/stream17"
  [[ -f "${body}/_SUCCESS" ]]
  [[ -f "${body}/proposal_graphs.pt" ]]
  mkdir -p "${output}"
  "${PYTHON}" "${SOURCE}/scripts/shard_proposal_graphs.py" \
    --proposal-graphs "${body}/proposal_graphs.pt" \
    --output-dir "${output}/proposal_shards" --shard-count 4
  local pids=() rank gpu worker
  for rank in 0 1 2 3; do
    if (( rank % 2 == 0 )); then gpu="${gpu0}"; else gpu="${gpu1}"; fi
    worker="${output}/workers/worker-${rank}"
    mkdir -p "${worker}"
    env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
      PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
      "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
      --proposal-graphs "${output}/proposal_shards/shard-$(printf '%03d' "${rank}").pt" \
      --checkpoint "${REFINER}" --output-dir "${worker}" \
      --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${REFINER_SEED}" \
      --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
      >"${RUN}/logs/${mode}_worker${rank}.out" \
      2>"${RUN}/logs/${mode}_worker${rank}.err" &
    pids+=("$!")
  done
  local worker_status=0 pid
  for pid in "${pids[@]}"; do wait "${pid}" || worker_status="$?"; done
  (( worker_status == 0 ))
  "${PYTHON}" "${SOURCE}/scripts/merge_refinement_shards.py" \
    --proposal-graphs "${body}/proposal_graphs.pt" \
    --worker-dir "${output}/workers/worker-0" \
    --worker-dir "${output}/workers/worker-1" \
    --worker-dir "${output}/workers/worker-2" \
    --worker-dir "${output}/workers/worker-3" \
    --output-dir "${output}/refine" --diff-steps 800
  local metrics result
  metrics="${output}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]]
  result="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
  [[ -f "${result}" ]]
  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${output}/refine" \
    --output-dir "${output}/generation" --arm candidate --repeat 0 \
    --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" \
    --denominator 256
  touch "${output}/_SUCCESS"
}

run_refine closure_control "${allocated[0]}" "${allocated[1]}" & control_pid="$!"
run_refine potential_closed "${allocated[2]}" "${allocated[3]}" & potential_pid="$!"
status=0
wait "${control_pid}" || status="$?"
wait "${potential_pid}" || status="$?"
(( status == 0 ))

"${PYTHON}" - "${RUN}" "${NATIVE_RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cells = []
for mode in ("closure_control", "potential_closed"):
    cell = root / mode / "stream17"
    metrics = json.load(open(cell / "refine" / "refinement_metrics.json"))
    rows = [json.loads(line) for line in open(cell / "generation" / "generation.jsonl") if line.strip()]
    assert len(rows) == 256
    cells.append({
        "mode": mode,
        "requested": 256,
        "reported": metrics.get("num_output_structures", metrics.get("output_count")),
        "generation_rows": len(rows),
        "world_size": metrics.get("world_size"),
    })
(root / "REFINEMENT_FINAL.json").write_text(json.dumps({
    "schema": "potential_closure_tau800_stream17_v1",
    "native_run": str(pathlib.Path(sys.argv[2]).resolve()),
    "stream": 17,
    "tau": 800,
    "seed_by_sample_index": True,
    "cells": cells,
}, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

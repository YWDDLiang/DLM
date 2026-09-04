#!/usr/bin/env bash
#SBATCH --job-name=spad-paper1000-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=spad-paper1000-eval-%j.out
#SBATCH --error=spad-paper1000-eval-%j.err

set -Eeuo pipefail
IFS=$'\n\t'

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly GENERATION_RUN="${SPAD_PAPER1000_GENERATION_RUN:?export completed wrapper218 run}"
readonly WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GLOBAL_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly OFFICIAL_CACHE="${ROOT}/runs/spad_basin_closure_official_20260904_v1/official_mp_cache"
readonly FINALIZER="${SOURCE}/scripts/finalize_spad_paper1000_official.py"
readonly PYTHON="${SPAD_BASIN_PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
readonly RUN="${ROOT}/runs/spad_basin_paper1000_eval_${SLURM_JOB_ID}"

[[ "${WORLD_SIZE}" =~ ^[1-4]$ ]]
for path in \
  "${GENERATION_RUN}/_SUCCESS" \
  "${GENERATION_RUN}/PAPER1000_GENERATION_FINAL.json" \
  "${GENERATION_RUN}/raw_generation/generation.jsonl" \
  "${GENERATION_RUN}/refined_generation/generation.jsonl" \
  "${GLOBAL_CACHE}" "${RESUMABLE}" "${FULL}" "${FINALIZER}" \
  "${OFFICIAL_CACHE}/completion_SUCCESS" \
  "${OFFICIAL_CACHE}/official_slim_cache.jsonl" \
  "${OFFICIAL_CACHE}/unresolved_chemsys.jsonl"; do
  [[ -f "${path}" ]]
done

mkdir "${RUN}"
mkdir "${RUN}/logs"
declare -a CHILD_PIDS=()
cleanup_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -0 "${pid}" 2>/dev/null && kill -TERM "${pid}" 2>/dev/null || true
  done
}
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  cleanup_children
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap 'cleanup_children; exit 130' INT TERM

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq "${WORLD_SIZE}" ]]

printf '%s\n' \
  $'field\tvalue' \
  $'schema\tspad_basin_paper1000_eval_v1' \
  $'conditional_valid_denominator\t1000' \
  $'endpoint_order\traw_then_tau800' \
  "world_size\t${WORLD_SIZE}" \
  $'direct\tfalse' \
  $'new_official_query\tfalse' \
  $'uncovered_official_cache\tretained_as_unknown_not_stable' \
  >"${RUN}/scientific_contract.tsv"

generation_path() {
  local endpoint="$1"
  printf '%s\n' "${GENERATION_RUN}/${endpoint}_generation/generation.jsonl"
}

run_endpoint() {
  local endpoint="$1" generation cell rank pid status=0
  local -a shard_pids=()
  generation="$(generation_path "${endpoint}")"
  cell="${RUN}/${endpoint}"
  mkdir -p "${cell}/input" "${cell}/shards"
  env CUDA_VISIBLE_DEVICES= PYTHONPATH="${R03E}/runtime" "${PYTHON}" \
    - "${generation}" "${cell}/input" <<'PY'
import pathlib
import sys
from crystal_dlm.wqcodiff.crysllmgen import a100_sun

output = pathlib.Path(sys.argv[2])
manifest = a100_sun.prepare_a100_input(
    generation_jsonl=pathlib.Path(sys.argv[1]),
    output_pt=output / "all_attempts.pt",
    output_manifest=output / "input_manifest.json",
    expected_attempts=1000,
)
assert manifest["total_attempts"] == 1000
PY

  CHILD_PIDS=()
  for ((rank=0; rank<WORLD_SIZE; rank++)); do
    env CUDA_VISIBLE_DEVICES="${allocated_gpus[${rank}]}" \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
      "${PYTHON}" "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
      --generation-jsonl "${generation}" \
      --generated-pt "${cell}/input/all_attempts.pt" \
      --input-manifest "${cell}/input/input_manifest.json" \
      --base-cache "${GLOBAL_CACHE}" --frozen-resumable-py "${RESUMABLE}" \
      --output-jsonl "${cell}/shards/cache_rank${rank}.jsonl" \
      --report-json "${cell}/shards/report_rank${rank}.json" \
      --denominator 1000 --shard-rank "${rank}" \
      --shard-count "${WORLD_SIZE}" --device cuda \
      >"${RUN}/logs/${endpoint}_relax_rank${rank}.out" \
      2>"${RUN}/logs/${endpoint}_relax_rank${rank}.err" &
    shard_pids+=("$!")
    CHILD_PIDS+=("$!")
  done
  for pid in "${shard_pids[@]}"; do wait "${pid}" || status=1; done
  CHILD_PIDS=()
  [[ "${status}" -eq 0 ]]

  cp --reflink=auto "${GLOBAL_CACHE}" "${cell}/working_relax_cache.jsonl"
  for ((rank=0; rank<WORLD_SIZE; rank++)); do
    cat "${cell}/shards/cache_rank${rank}.jsonl" \
      >>"${cell}/working_relax_cache.jsonl"
  done
  env CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=1000 \
    PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${generation}" --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/working_relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/${endpoint}_full.out" \
    2>"${RUN}/logs/${endpoint}_full.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
}

# Raw is the primary native endpoint and the slower relaxation.  Give it every
# allocated GPU first, then reuse every GPU for the already-refined endpoint.
run_endpoint raw
run_endpoint refined

finalize_endpoint() {
  local endpoint="$1" generation cell
  local -a model_args=()
  generation="$(generation_path "${endpoint}")"
  cell="${RUN}/${endpoint}"
  [[ "${endpoint}" == raw ]] || model_args=(--model494 --model494-tau 800)
  env CUDA_VISIBLE_DEVICES= H1_ACTIVE_DENOMINATOR=1000 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/eval_runtime" \
    "${PYTHON}" "${FINALIZER}" \
    --preofficial-labels \
      "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" \
    --generation-jsonl "${generation}" --official-cache-dir "${OFFICIAL_CACHE}" \
    --endpoint "paper1000_${endpoint}" \
    ${model_args[@]+"${model_args[@]}"} --output-dir "${cell}/official" \
    >"${RUN}/logs/${endpoint}_official.out" \
    2>"${RUN}/logs/${endpoint}_official.err"
  [[ -f "${cell}/official/_SUCCESS" ]]
}

finalize_endpoint raw
finalize_endpoint refined

"${PYTHON}" - "${RUN}" "${GENERATION_RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]); generation = pathlib.Path(sys.argv[2]).resolve()
cells = []
for endpoint in ("raw", "refined"):
    report = json.loads(
        (root / endpoint / "official/PAPER1000_OFFICIAL_FINAL.json").read_text()
    )
    assert report["conditional_parser_valid_denominator"] == 1000
    counts = report["counts"]
    cells.append(
        {
            "endpoint": endpoint,
            "strict_sun": int(counts["strict_sun"]),
            "meta_sun": int(counts["meta_sun"]),
            "strict_stable": int(counts["strict_stable"]),
            "meta_stable": int(counts["meta_stable"]),
            "novel_unique": int(counts["novel_unique"]),
            "hull_known": int(counts["hull_known"]),
            "hull_unknown": int(counts["hull_unknown"]),
            "model494": endpoint == "refined",
        }
    )
final = {
    "schema": "spad_basin_paper1000_final_v1",
    "generation_run": str(generation),
    "conditional_parser_valid_denominator": 1000,
    "cells": cells,
    "target_counts": {"strict": 100, "meta": 500},
    "target_met_by_endpoint": {
        cell["endpoint"]: cell["strict_sun"] >= 100 and cell["meta_sun"] >= 500
        for cell in cells
    },
    "direct_run": False,
    "new_official_query": False,
    "outcome_selection_or_replacement": False,
}
(root / "PAPER1000_FINAL.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(final, sort_keys=True))
PY

trap - ERR INT TERM
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

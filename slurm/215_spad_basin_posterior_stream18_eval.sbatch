#!/usr/bin/env bash
#SBATCH --job-name=spad-post-s18-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=440G
#SBATCH --time=10:00:00
#SBATCH --output=spad-post-s18-eval-%j.out
#SBATCH --error=spad-post-s18-eval-%j.err

set -Eeuo pipefail
IFS=$'\n\t'

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly GENERATION_RUN="${SPAD_BASIN_POSTERIOR_GENERATION_RUN:?export the completed 214 generation run}"
readonly SELECTED_TAU="${SPAD_SELECTED_TAU:?export the same preregistered tau used by 214}"
readonly PRIMARY_ONLY="${SPAD_PRIMARY_ONLY:-false}"
readonly PRIMARY_WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"
readonly PLAN_DIR="${SPAD_PLAN_DIR:-${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903}"
readonly PLANNER_SEED="${SPAD_PLANNER_SEED:-23}"
readonly STREAM="${SPAD_EVAL_STREAM:-18}"
readonly DLM_SEED="${SPAD_DLM_SEED:-92117}"
readonly REFINER_SEED="${SPAD_REFINER_SEED:-102117}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GLOBAL_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly OFFICIAL_CACHE="${ROOT}/runs/spad_basin_closure_official_20260904_v1/official_mp_cache"
readonly S17_DIAGNOSTIC="${ROOT}/runs/spad_basin_closure_official_final_20260904_v1/attempt_results_official.jsonl"
readonly RAW_VALIDITY="${ROOT}/runs/spad_basin_closure_native_raw_screen_39710/screen/RAW_SCREEN_FINAL.json"
readonly FINALIZER="${SOURCE}/scripts/finalize_spad_basin_closure_official.py"
readonly PYTHON="${SPAD_BASIN_PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
readonly RUN="${ROOT}/runs/spad_basin_posterior_stream${STREAM}_eval_${SLURM_JOB_ID}"

case "${SELECTED_TAU}" in
  200|400|600|800) ;;
  *)
    printf 'SPAD_SELECTED_TAU must be one of 200, 400, 600, or 800; observed %s\n' \
      "${SELECTED_TAU}" >&2
    exit 2
    ;;
esac
[[ "${PRIMARY_ONLY}" == true || "${PRIMARY_ONLY}" == false ]] || {
  printf 'SPAD_PRIMARY_ONLY must be true or false\n' >&2
  exit 2
}
[[ "${PRIMARY_WORLD_SIZE}" =~ ^[1-4]$ ]] || {
  printf 'SPAD_PRIMARY_WORLD_SIZE must lie in [1,4]\n' >&2
  exit 2
}
for numeric_name in PLANNER_SEED STREAM DLM_SEED REFINER_SEED; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[0-9]+$ ]] || {
    printf '%s must be a non-negative integer; observed %s\n' \
      "${numeric_name}" "${numeric_value}" >&2
    exit 2
  }
done

mkdir "${RUN}"
mkdir "${RUN}/logs"

declare -a CHILD_PIDS=()
cleanup_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
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

for path in \
  "${GENERATION_RUN}/_SUCCESS" \
  "${GENERATION_RUN}/GENERATION_FINAL.json" \
  "${PLAN_DIR}/_SUCCESS" \
  "${PLAN_DIR}/manifest.json" \
  "${PLAN_DIR}/plans_for_dlm.jsonl" \
  "${PLAN_DIR}/ledger.jsonl" \
  "${GLOBAL_CACHE}" \
  "${RESUMABLE}" \
  "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
  "${FULL}" \
  "${FINALIZER}" \
  "${OFFICIAL_CACHE}/completion_SUCCESS" \
  "${OFFICIAL_CACHE}/official_slim_cache.jsonl" \
  "${OFFICIAL_CACHE}/unresolved_chemsys.jsonl" \
  "${S17_DIAGNOSTIC}" \
  "${RAW_VALIDITY}"; do
  [[ -f "${path}" ]] || {
    printf 'required input is missing: %s\n' "${path}" >&2
    exit 2
  }
done

"${PYTHON}" - "${GENERATION_RUN}" "${SELECTED_TAU}" "${PRIMARY_ONLY}" \
  "${PLAN_DIR}" "${PLANNER_SEED}" "${STREAM}" "${DLM_SEED}" \
  "${REFINER_SEED}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
tau = int(sys.argv[2])
primary_only = sys.argv[3] == "true"
plan_dir = pathlib.Path(sys.argv[4]).resolve(strict=True)
planner_seed = int(sys.argv[5])
stream = int(sys.argv[6])
dlm_seed = int(sys.argv[7])
refiner_seed = int(sys.argv[8])
report = json.loads((root / "GENERATION_FINAL.json").read_text(encoding="utf-8"))
assert report["schema"] == f"spad_basin_posterior_heldout_stream{stream}_generation_v1"
assert pathlib.Path(report["frozen_plan_dir"]).resolve(strict=True) == plan_dir
assert report["planner_sampling_seed"] == planner_seed
assert report["stream"] == stream
assert report["dlm_seed"] == dlm_seed
assert report["refiner_seed"] == refiner_seed
assert report["selected_tau"] == tau
assert report["primary_arm"] == "k10"
assert report["one_plan_one_trajectory"] is True
assert report["all_failures_retained_in_fixed_denominator"] is True
assert report["fixed_denominator"] == 256
assert report["automatic_route_tau_seed_checkpoint_choice"] is False
assert report["retry_rerank_replacement"] is False
expected_arms = ["k10"] if primary_only else ["closure_ce", "k10"]
assert report["primary_only"] is primary_only
assert [cell["arm"] for cell in report["cells"]] == expected_arms
for arm in expected_arms:
    for endpoint in ("raw", "refined"):
        path = root / arm / f"stream{stream}" / f"{endpoint}_generation" / "generation.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        assert len(rows) == 256
        assert [int(row["ordinal"]) for row in rows] == list(range(256))
        assert all(row["retry_or_replacement_used"] is False for row in rows)
        if endpoint == "raw":
            assert all(
                row["diffusion_refinement_applied"] is False
                and row["diffusion_refinement_steps"] == 0
                for row in rows
            )
        else:
            assert all(row["diffusion_refinement_steps"] in (None, tau) for row in rows)
PY

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
expected_gpus=4
[[ "${PRIMARY_ONLY}" == true ]] && expected_gpus="${PRIMARY_WORLD_SIZE}"
[[ "${#allocated_gpus[@]}" -eq "${expected_gpus}" ]] || {
  printf 'expected %s allocated GPUs, observed %s\n' \
    "${expected_gpus}" "${#allocated_gpus[@]}" >&2
  exit 2
}

printf '%s\n' \
  $'field\tvalue' \
  $'schema\t'"spad_basin_posterior_stream${STREAM}_eval_contract_v1" \
  "allocation\t${expected_gpus}_A800" \
  $'planner_seed\t'"${PLANNER_SEED}" \
  $'stream\t'"${STREAM}" \
  $'dlm_seed\t'"${DLM_SEED}" \
  $'refiner_seed\t'"${REFINER_SEED}" \
  "cells\t$([[ "${PRIMARY_ONLY}" == true ]] && printf k10_raw_refined || printf closure_ce_k10_x_raw_refined)" \
  "first_wave\t$([[ "${PRIMARY_ONLY}" == true ]] && printf 2_primary_cells || printf 4_cells)" \
  "selected_tau\t${SELECTED_TAU}" \
  $'primary_cell\tk10_refined' \
  $'strict_sun_target\t26_of_256' \
  $'meta_sun_target\t128_of_256' \
  $'automatic_result_choice\tfalse' \
  $'new_official_request\tfalse' \
  >"${RUN}/scientific_contract.tsv"

generation_path() {
  local arm="$1" endpoint="$2"
  printf '%s\n' \
    "${GENERATION_RUN}/${arm}/stream${STREAM}/${endpoint}_generation/generation.jsonl"
}

run_full_cell() {
  local slot="$1" arm="$2" endpoint="$3" cell generation
  cell="${RUN}/${arm}/${endpoint}"
  generation="$(generation_path "${arm}" "${endpoint}")"
  mkdir -p "${cell}"
  cp --reflink=auto "${GLOBAL_CACHE}" "${cell}/working_relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${allocated_gpus[${slot}]}" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${generation}" --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/working_relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/${arm}_${endpoint}_full.out" \
    2>"${RUN}/logs/${arm}_${endpoint}_full.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
  [[ -f "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" ]]
}

run_primary_endpoint_sharded() {
  local endpoint="$1" first_slot="$2" shard_count="$3" arm=k10
  local cell generation rank slot pid status=0
  local -a shard_pids=()
  cell="${RUN}/${arm}/${endpoint}"
  generation="$(generation_path "${arm}" "${endpoint}")"
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
    expected_attempts=256,
)
assert manifest["total_attempts"] == 256
PY

  for ((rank=0; rank<shard_count; rank++)); do
    slot=$((first_slot + rank))
    env CUDA_VISIBLE_DEVICES="${allocated_gpus[${slot}]}" \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
      "${PYTHON}" "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
      --generation-jsonl "${generation}" \
      --generated-pt "${cell}/input/all_attempts.pt" \
      --input-manifest "${cell}/input/input_manifest.json" \
      --base-cache "${GLOBAL_CACHE}" --frozen-resumable-py "${RESUMABLE}" \
      --output-jsonl "${cell}/shards/cache_rank${rank}.jsonl" \
      --report-json "${cell}/shards/report_rank${rank}.json" \
      --denominator 256 --shard-rank "${rank}" \
      --shard-count "${shard_count}" --device cuda \
      >"${RUN}/logs/k10_${endpoint}_relax_rank${rank}.out" \
      2>"${RUN}/logs/k10_${endpoint}_relax_rank${rank}.err" &
    shard_pids+=("$!")
  done
  for pid in "${shard_pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ "${status}" -eq 0 ]]

  cp --reflink=auto "${GLOBAL_CACHE}" "${cell}/working_relax_cache.jsonl"
  for ((rank=0; rank<shard_count; rank++)); do
    cat "${cell}/shards/cache_rank${rank}.jsonl" \
      >>"${cell}/working_relax_cache.jsonl"
  done
  env CUDA_VISIBLE_DEVICES="${allocated_gpus[${first_slot}]}" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${generation}" --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/working_relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/k10_${endpoint}_full.out" \
    2>"${RUN}/logs/k10_${endpoint}_full.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
  [[ -f "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" ]]
}

run_wave_one() {
  CHILD_PIDS=()
  if [[ "${PRIMARY_ONLY}" == true ]]; then
    # Raw relaxation is much slower than the already-refined endpoint.  Give
    # every allocated GPU to raw first, then reuse all ranks for refined.
    run_primary_endpoint_sharded raw 0 "${expected_gpus}"
    run_primary_endpoint_sharded refined 0 "${expected_gpus}"
    return
  else
    run_full_cell 0 closure_ce raw & CHILD_PIDS+=("$!")
    run_full_cell 1 closure_ce refined & CHILD_PIDS+=("$!")
    run_full_cell 2 k10 raw & CHILD_PIDS+=("$!")
    run_full_cell 3 k10 refined & CHILD_PIDS+=("$!")
  fi
  local status=0 pid
  for pid in "${CHILD_PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  [[ "${status}" -eq 0 ]]
}

run_wave_one

finalize_cell() {
  local arm="$1" endpoint="$2" cell generation
  local -a model_args=()
  cell="${RUN}/${arm}/${endpoint}"
  generation="$(generation_path "${arm}" "${endpoint}")"
  if [[ "${endpoint}" == refined ]]; then
    model_args=(--model494 --model494-tau "${SELECTED_TAU}")
  fi
  env CUDA_VISIBLE_DEVICES= H1_ACTIVE_DENOMINATOR=256 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/eval_runtime" \
    "${PYTHON}" "${FINALIZER}" \
    --preofficial-labels \
      "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" \
    --generation-jsonl "${generation}" \
    --official-cache-dir "${OFFICIAL_CACHE}" \
    --baseline-official-attempts "${S17_DIAGNOSTIC}" \
    --raw-screen-json "${RAW_VALIDITY}" --raw-screen-arm closure_ce \
    --endpoint "${arm}_${endpoint}_stream${STREAM}" \
    ${model_args[@]+"${model_args[@]}"} --output-dir "${cell}/official" \
    >"${RUN}/logs/${arm}_${endpoint}_official.out" \
    2>"${RUN}/logs/${arm}_${endpoint}_official.err"
  [[ -f "${cell}/official/_SUCCESS" ]]
}

if [[ "${PRIMARY_ONLY}" == true ]]; then
  finalize_cell k10 raw
  finalize_cell k10 refined
else
  for arm in closure_ce k10; do
    finalize_cell "${arm}" raw
    finalize_cell "${arm}" refined
  done
fi

"${PYTHON}" - "${RUN}" "${GENERATION_RUN}" "${SELECTED_TAU}" \
  "${OFFICIAL_CACHE}" "${S17_DIAGNOSTIC}" "${PRIMARY_ONLY}" \
  "${PLAN_DIR}" "${PLANNER_SEED}" "${STREAM}" "${DLM_SEED}" \
  "${REFINER_SEED}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
generation_run = pathlib.Path(sys.argv[2]).resolve(strict=True)
tau = int(sys.argv[3])
official_cache = pathlib.Path(sys.argv[4]).resolve(strict=True)
s17_diagnostic = pathlib.Path(sys.argv[5]).resolve(strict=True)
primary_only = sys.argv[6] == "true"
plan_dir = pathlib.Path(sys.argv[7]).resolve(strict=True)
planner_seed = int(sys.argv[8])
stream = int(sys.argv[9])
dlm_seed = int(sys.argv[10])
refiner_seed = int(sys.argv[11])

cells = []
by_name = {}
cell_names = (("k10", "raw"), ("k10", "refined")) if primary_only else tuple(
    (arm, endpoint)
    for arm in ("closure_ce", "k10")
    for endpoint in ("raw", "refined")
)
for arm, endpoint in cell_names:
        report = json.loads(
            (root / arm / endpoint / "official" / "BASIN_CLOSURE_OFFICIAL_FINAL.json")
            .read_text(encoding="utf-8")
        )
        assert report["fixed_denominator"] == 256
        assert report["endpoint"] == f"{arm}_{endpoint}_stream{stream}"
        assert report["selection_retry_replacement"] is False
        if endpoint == "raw":
            assert report["model494"] is False
            assert report["model494_tau"] is None
        else:
            assert report["model494"] is True
            assert report["model494_tau"] == tau
        counts = report["counts"]
        rates = report["rates_all_attempts"]
        summary = {
            "arm": arm,
            "endpoint": endpoint,
            "primary": arm == "k10" and endpoint == "refined",
            "model494": endpoint == "refined",
            "model494_tau": tau if endpoint == "refined" else None,
            "fixed_denominator": 256,
            "reconstructed": {
                "count": int(counts["reconstructed"]),
                "rate": int(counts["reconstructed"]) / 256,
            },
            "NU": {
                "count": int(counts["novel_unique"]),
                "rate": int(counts["novel_unique"]) / 256,
            },
            "hull_known": {
                "count": int(counts["hull_known"]),
                "rate": int(counts["hull_known"]) / 256,
            },
            "strict_stable": {
                "count": int(counts["strict_stable"]),
                "rate": float(rates["strict_stable"]),
            },
            "meta_stable": {
                "count": int(counts["meta_stable"]),
                "rate": float(rates["meta_stable"]),
            },
            "strict_sun": {
                "count": int(counts["strict_sun"]),
                "rate": float(rates["strict_sun"]),
            },
            "meta_sun": {
                "count": int(counts["meta_sun"]),
                "rate": float(rates["meta_sun"]),
            },
        }
        cells.append(summary)
        by_name[(arm, endpoint)] = summary

primary = by_name[("k10", "refined")]
strict_met = primary["strict_sun"]["count"] >= 26
meta_met = primary["meta_sun"]["count"] >= 128
final = {
    "schema": f"spad_basin_posterior_heldout_stream{stream}_final_v1",
    "generation_run": str(generation_run),
    "frozen_plan_dir": str(plan_dir),
    "planner_sampling_seed": planner_seed,
    "stream": stream,
    "dlm_seed": dlm_seed,
    "refiner_seed": refiner_seed,
    "selected_tau": tau,
    "selected_tau_source": "completed_stream17_development_calibration",
    "fixed_denominator": 256,
    "primary_cell": "k10_refined",
    "primary_only": primary_only,
    "cells": cells,
    "target": {
        "strict_sun_min_count": 26,
        "strict_sun_min_rate": 0.10,
        "meta_sun_min_count": 128,
        "meta_sun_min_rate": 0.50,
    },
    "target_met": strict_met and meta_met,
    "target_met_by_metric": {
        "strict_sun": strict_met,
        "meta_sun": meta_met,
    },
    "absolute_heldout_counts_and_rates_are_primary": True,
    "paired_s17_closure_comparison_is_diagnostic_only": True,
    "s17_diagnostic_attempts": str(s17_diagnostic),
    "reused_official_cache": str(official_cache),
    "new_official_request": False,
    "automatic_route_tau_seed_checkpoint_choice": False,
    "all_failures_retained_in_fixed_denominator": True,
}
(root / f"HELDOUT_STREAM{stream}_FINAL.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(final, sort_keys=True))
PY

trap - ERR INT TERM
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

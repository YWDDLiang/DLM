#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-full-sun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=pcdlm-full-sun-%j.out
#SBATCH --error=pcdlm-full-sun-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly NATIVE_RUN="${POTENTIAL_CLOSURE_NATIVE_RUN:?}"
readonly TAU800_RUN="${POTENTIAL_CLOSURE_TAU800_RUN:?}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_full_sun_preofficial_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${NATIVE_RUN}/_SUCCESS" ]]
[[ -f "${TAU800_RUN}/_SUCCESS" ]]
[[ -f "${CACHE}" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu[@]}" -eq 4 ]]

generation_path() {
  local mode="$1" endpoint="$2"
  if [[ "${endpoint}" == raw ]]; then
    printf '%s\n' "${NATIVE_RUN}/${mode}/stream17/raw_generation/generation.jsonl"
  else
    printf '%s\n' "${TAU800_RUN}/${mode}/stream17/generation/generation.jsonl"
  fi
}

run_full() {
  local slot="$1" mode="$2" endpoint="$3" generation cell
  generation="$(generation_path "${mode}" "${endpoint}")"
  cell="${RUN}/cells/${mode}_${endpoint}_s17"
  [[ -f "${generation}" ]]
  [[ "$(wc -l <"${generation}")" -eq 256 ]]
  mkdir -p "${cell}/evaluation"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${generation}" --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" --working-relax-cache "${cell}/relax_cache.jsonl" \
    --output-dir "${cell}/evaluation/full_reconstructed" \
    >"${RUN}/logs/${mode}_${endpoint}.out" \
    2>"${RUN}/logs/${mode}_${endpoint}.err"
}

run_full 0 closure_control raw & p0="$!"
run_full 1 potential_closed raw & p1="$!"
run_full 2 closure_control tau800 & p2="$!"
run_full 3 potential_closed tau800 & p3="$!"
status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || status="$?"; done
(( status == 0 ))

"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cells = []
for mode in ("closure_control", "potential_closed"):
    for endpoint in ("raw", "tau800"):
        path = root / "cells" / f"{mode}_{endpoint}_s17" / "evaluation" / "full_reconstructed"
        assert (path / "_SUCCESS").is_file()
        report = json.load(open(path / "report.json"))
        assert report["raw_attempts"] == 256
        cells.append({
            "mode": mode,
            "endpoint": endpoint,
            "attempts": report["raw_attempts"],
            "reconstructed": report["reconstructed"],
            "novel_unique": report["novel_unique"],
            "chgnet_relaxation_known": report["chgnet_relaxation_known"],
        })
(root / "PREOFFICIAL_FINAL.json").write_text(json.dumps({
    "schema": "potential_closure_full_sun_preofficial_v1",
    "cells": cells,
    "direct_run": False,
}, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

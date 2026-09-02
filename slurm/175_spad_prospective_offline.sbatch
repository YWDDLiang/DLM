#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=spad-prosp-eval-%j.out
#SBATCH --error=spad-prosp-eval-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly GENERATION_RUN="${SPAD_PROSPECTIVE_GENERATION_RUN:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_offline_${SLURM_JOB_ID}"
readonly EXIT_GRACE=300
readonly TERM_GRACE=30
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${GENERATION_RUN}/_SUCCESS" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu[@]}" -eq 6 ]]

for arm in B0 BC BS; do
  for stream in 17 18; do
    for endpoint in raw refined; do
      source_generation="${GENERATION_RUN}/${arm}/stream${stream}/${endpoint}_generation"
      cell="${RUN}/${arm}/stream${stream}/${endpoint}"
      [[ -f "${source_generation}/_SUCCESS" ]]
      mkdir -p "${cell}"
      cp -a "${source_generation}" "${cell}/generation"
      [[ "$(wc -l <"${cell}/generation/generation.jsonl")" -eq 256 ]]
    done
  done
done

process_alive() {
  kill -0 -- "-$1" 2>/dev/null || kill -0 "$1" 2>/dev/null
}

wait_completed() {
  local pid="$1" artifact="$2" seen=-1 deadline rc=0
  while process_alive "${pid}"; do
    if [[ -e "${artifact}" ]]; then
      (( seen < 0 )) && seen="${SECONDS}"
      if (( SECONDS - seen >= EXIT_GRACE )); then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        deadline=$((SECONDS + TERM_GRACE))
        while process_alive "${pid}" && (( SECONDS < deadline )); do sleep 2; done
        process_alive "${pid}" && { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true; }
        break
      fi
    fi
    sleep 2
  done
  wait "${pid}" || rc="$?"
  [[ -e "${artifact}" ]] || return 1
  (( rc == 0 )) || return "${rc}"
}

run_direct() {
  local arm="$1" stream="$2" endpoint="$3"
  local cell="${RUN}/${arm}/stream${stream}/${endpoint}"
  mkdir -p "${cell}/evaluation"
  setsid env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${DIRECT}" --generation-jsonl "${cell}/generation/generation.jsonl" \
    --gt-csv "${GT}" --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell}/evaluation/direct" --eval-model-name mp20 \
    >"${RUN}/logs/${arm}_${stream}_${endpoint}_direct.out" \
    2>"${RUN}/logs/${arm}_${stream}_${endpoint}_direct.err" &
  local pid="$!"
  wait_completed "${pid}" "${cell}/evaluation/direct/report.json"
}

run_full() {
  local slot="$1" arm="$2" stream="$3" endpoint="$4"
  local cell="${RUN}/${arm}/stream${stream}/${endpoint}"
  mkdir -p "${cell}/evaluation"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  setsid env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" OPENBLAS_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${cell}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/relax_cache.jsonl" \
    --output-dir "${cell}/evaluation/full_reconstructed" \
    >"${RUN}/logs/${arm}_${stream}_${endpoint}_full.out" \
    2>"${RUN}/logs/${arm}_${stream}_${endpoint}_full.err" &
  local pid="$!"
  wait_completed "${pid}" "${cell}/evaluation/full_reconstructed/_SUCCESS"
}

run_cell() {
  local slot="$1" arm="$2" stream="$3" endpoint="$4"
  local direct_pid
  run_direct "${arm}" "${stream}" "${endpoint}" & direct_pid="$!"
  run_full "${slot}" "${arm}" "${stream}" "${endpoint}"
  wait "${direct_pid}"
}

run_wave() {
  local endpoint="$1" index=0 status=0 pid
  local -a pids=()
  for arm in B0 BC BS; do
    for stream in 17 18; do
      run_cell "${index}" "${arm}" "${stream}" "${endpoint}" &
      pids+=("$!")
      index=$((index + 1))
    done
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status="$?"; done
  return "${status}"
}

run_wave raw
run_wave refined

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cells = []
for arm in ("B0", "BC", "BS"):
    for stream in (17, 18):
        for endpoint in ("raw", "refined"):
            cell = root / arm / f"stream{stream}" / endpoint / "evaluation"
            direct = json.load(open(cell / "direct/report.json", encoding="utf-8"))
            full = cell / "full_reconstructed"
            for required in ("_SUCCESS", "attempt_labels_preofficial.jsonl", "worker_cleanup.json"):
                assert (full / required).is_file(), (arm, stream, endpoint, required)
            cells.append({
                "arm": arm,
                "stream": stream,
                "endpoint": endpoint,
                "requested": 256,
                "composition_valid": direct["comp_valid_count"],
                "structure_valid": direct["struct_valid_count"],
                "direct_joint": direct["valid_count"],
            })
report = {"schema": "spad_prospective_offline_v1", "cells": cells, "official": False}
(root / "OFFLINE_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"

#!/usr/bin/env bash
#SBATCH --job-name=spad-response-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=spad-response-eval-%j.out
#SBATCH --error=spad-response-eval-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly GENERATION="${SPAD_RESPONSE_GENERATION_RUN:?}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_response_fast_eval_${SLURM_JOB_ID}"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${GENERATION}/_SUCCESS" && -f "${GENERATION}/GENERATION_FINAL.json" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

"${PYTHON}" - "${GENERATION}/GENERATION_FINAL.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["schema"] == "spad_model494_response_followup_v1"
assert report["same_frozen_plans_and_source_bs_trajectories"] is True
assert report["one_deterministic_response_pass"] is True
assert report["candidate_selection_rerank_replacement"] is False
assert report["direct_evaluation"] == "DEFERRED_COST"
assert {int(cell["stream"]) for cell in report["cells"]} == {17, 18}
assert all(int(cell["requested"]) == int(cell["guided"]) == 256 for cell in report["cells"])
PY

run_endpoint() {
  local stream="$1" endpoint="$2" gpu="$3" source_generation cell
  source_generation="${GENERATION}/stream${stream}/${endpoint}_generation"
  cell="${RUN}/stream${stream}/${endpoint}"
  [[ -f "${source_generation}/_SUCCESS" ]]
  mkdir -p "${cell}"
  cp -a "${source_generation}" "${cell}/generation"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  printf '%s\n' 'DEFERRED_COST' >"${cell}/DIRECT_STATUS"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${cell}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/stream${stream}_${endpoint}_chgnet.out" \
    2>"${RUN}/logs/stream${stream}_${endpoint}_chgnet.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
  [[ -f "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" ]]
  [[ -f "${cell}/full_reconstructed/worker_cleanup.json" ]]
}

run_stream() {
  local stream="$1" gpu="$2"
  run_endpoint "${stream}" raw "${gpu}"
  run_endpoint "${stream}" refined "${gpu}"
}

run_stream 17 "${allocated[0]}" & pid17="$!"
run_stream 18 "${allocated[1]}" & pid18="$!"
status=0
wait "${pid17}" || status="$?"
wait "${pid18}" || status="$?"
(( status == 0 ))

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cells = []
for stream in (17, 18):
    for endpoint in ("raw", "refined"):
        cell = root / f"stream{stream}" / endpoint
        labels = [
            json.loads(line)
            for line in open(cell / "full_reconstructed/attempt_labels_preofficial.jsonl", encoding="utf-8")
            if line.strip()
        ]
        assert len(labels) == 256
        assert (cell / "DIRECT_STATUS").read_text().strip() == "DEFERRED_COST"
        cells.append({
            "stream": stream,
            "endpoint": endpoint,
            "requested": 256,
            "reconstructed": sum(row.get("reconstructed") is True for row in labels),
            "novel": sum(row.get("novel") is True for row in labels),
            "unique": sum(row.get("unique_representative") is True for row in labels),
            "novel_unique": sum(row.get("novel_unique") is True for row in labels),
            "chgnet_energy_known": sum(row.get("chgnet_energy_per_atom") is not None for row in labels),
            "direct": "DEFERRED_COST",
        })
report = {
    "schema": "spad_response_fast_eval_v1",
    "requested_per_cell": 256,
    "cells": cells,
    "direct": "DEFERRED_COST",
    "official_query": False,
    "same_frozen_compositions": True,
}
(root / "FAST_EVAL_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"

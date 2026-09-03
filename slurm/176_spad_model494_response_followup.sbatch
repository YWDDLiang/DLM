#!/usr/bin/env bash
#SBATCH --job-name=spad-response
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=spad-response-%j.out
#SBATCH --error=spad-response-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_model494_response_followup_${SLURM_JOB_ID}"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

source_run() {
  case "$1" in
    17) printf '%s\n' "${ROOT}/runs/spad_prospective_generation_s17_39537" ;;
    18) printf '%s\n' "${ROOT}/runs/spad_prospective_generation_s18_39538" ;;
    *) return 2 ;;
  esac
}

dlm_seed() {
  [[ "$1" == 17 ]] && printf '91117\n' || printf '92117\n'
}

refiner_seed() {
  [[ "$1" == 17 ]] && printf '101117\n' || printf '102117\n'
}

run_stream() {
  local stream="$1" gpu="$2" source cell metrics output_file seed rseed
  source="$(source_run "${stream}")"
  cell="${RUN}/stream${stream}"
  seed="$(dlm_seed "${stream}")"
  rseed="$(refiner_seed "${stream}")"
  [[ -f "${source}/_SUCCESS" ]]
  [[ -f "${source}/BS/stream${stream}/body/_SUCCESS" ]]
  [[ -f "${source}/BS/stream${stream}/refine/refinement_metrics.json" ]]
  mkdir "${cell}"

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/apply_model494_response_backfill.py" \
    --model-path "${MODEL}" --checkpoint-path "${BS_POLICY}" \
    --body-rows "${source}/BS/stream${stream}/body/raw_generations.jsonl" \
    --body-proposal-graphs "${source}/BS/stream${stream}/body/proposal_graphs.pt" \
    --refinement-metrics "${source}/BS/stream${stream}/refine/refinement_metrics.json" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" --batch-size 8 \
    >"${RUN}/logs/stream${stream}_response.out" \
    2>"${RUN}/logs/stream${stream}_response.err"
  [[ -f "${cell}/body/_SUCCESS" ]]

  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body" --output-dir "${cell}/raw_generation" \
    --seed "${stream}" --dlm-seed "${seed}" --denominator 256 \
    --source-arm full_axis --arm-label candidate \
    --method-label SPAD-MODEL494-RESPONSE-RAW \
    --planner-arm-label C3FD-Llama-pointer-frozen \
    --body-arm-label BS-response-aligned \
    --schedule-arm-label SPAD-reverse-all-site-response

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${cell}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${cell}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${rseed}" \
    --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${RUN}/logs/stream${stream}_refine.out" \
    2>"${RUN}/logs/stream${stream}_refine.err"
  metrics="${cell}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]]
  output_file="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]]

  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${cell}/body" --refine-dir "${cell}/refine" \
    --output-dir "${cell}/refined_generation" --arm candidate --repeat 0 \
    --dlm-seed "${seed}" --refiner-seed "${rseed}" --denominator 256 \
    --refinement-steps 800 --invalid-refined-as-failure \
    --method-label SPAD-MODEL494-RESPONSE-REFINED \
    --planner-arm-label C3FD-Llama-pointer-frozen \
    --body-arm-label BS-response-aligned \
    --schedule-arm-label SPAD-reverse-all-site-response
  touch "${cell}/_SUCCESS"
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
    cell = root / f"stream{stream}"
    metrics = json.load(open(cell / "body/sample_metrics.json", encoding="utf-8"))
    refine = json.load(open(cell / "refine/refinement_metrics.json", encoding="utf-8"))
    raw = [json.loads(line) for line in open(cell / "raw_generation/generation.jsonl", encoding="utf-8")]
    refined = [json.loads(line) for line in open(cell / "refined_generation/generation.jsonl", encoding="utf-8")]
    assert metrics["requested_samples"] == metrics["guided_samples"] == 256
    assert metrics["unguided_samples"] == 0
    assert len(raw) == len(refined) == 256
    assert {int(row["sample_idx"]) for row in raw} == set(range(256))
    assert {int(row["sample_idx"]) for row in refined} == set(range(256))
    cells.append({
        "stream": stream,
        "requested": 256,
        "guided": metrics["guided_samples"],
        "body_parsed": metrics["parse_success"],
        "body_graphs": metrics["graph_success"],
        "changed_samples": metrics["changed_samples"],
        "changed_components": metrics["changed_components"],
        "guidance_skips": metrics["guidance_skip_reasons"],
        "refined": int(refine.get("num_proposals", 0)),
    })
report = {
    "schema": "spad_model494_response_followup_v1",
    "cells": cells,
    "same_frozen_plans_and_source_bs_trajectories": True,
    "one_deterministic_response_pass": True,
    "lattice_tokens_frozen": True,
    "response_not_claimed_force": True,
    "planner_resampling": False,
    "candidate_selection_rerank_replacement": False,
    "model494_tau": 800,
    "direct_evaluation": "DEFERRED_COST",
}
(root / "GENERATION_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"

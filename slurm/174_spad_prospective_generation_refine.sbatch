#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=spad-prosp-gen-%j.out
#SBATCH --error=spad-prosp-gen-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly PLAN_DIR="${SPAD_PROSPECTIVE_PLAN_DIR:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BASE_POLICY="${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392"
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_generation_${SLURM_JOB_ID}"
readonly POINTER_PLANS="${PLAN_DIR}/plans_for_dlm.jsonl"
readonly PLAN_LEDGER="${PLAN_DIR}/ledger.jsonl"
readonly CANONICAL_PLANS="${RUN}/canonical_program_plans.jsonl"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${PLAN_DIR}/_SUCCESS" && -f "${POINTER_PLANS}" && -f "${PLAN_LEDGER}" ]]
[[ -f "${BASE_POLICY}/adapter_model.safetensors" && -f "${BS_POLICY}/adapter_model.safetensors" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 6 ]]

"${PYTHON}" - "${POINTER_PLANS}" "${CANONICAL_PLANS}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert 244 <= len(rows) <= 256
assert len({int(row["sample_idx"]) for row in rows}) == len(rows)
assert all(0 <= int(row["sample_idx"]) < 256 for row in rows)
with open(sys.argv[2], "x", encoding="utf-8") as handle:
    for source in rows:
        row = dict(source)
        elements = list(row["plan_state"]["elements"])
        row["species_program"] = elements
        row["species_program_indices"] = list(range(len(elements)))
        row["species_program_source"] = "canonical_schedule_control"
        handle.write(json.dumps(row, sort_keys=True) + "\n")
PY

stream_seed() {
  [[ "$1" == 17 ]] && printf '91117\n' || printf '92117\n'
}

refiner_seed() {
  [[ "$1" == 17 ]] && printf '101117\n' || printf '102117\n'
}

run_cell() {
  local arm="$1" stream="$2" gpu="$3"
  local policy plans schedule seed rseed cell metrics result
  seed="$(stream_seed "${stream}")"
  rseed="$(refiner_seed "${stream}")"
  cell="${RUN}/${arm}/stream${stream}"
  mkdir -p "${cell}"
  local -a extras=(--canonicalize-periodic-alias --pbc-min-distance-mask)
  case "${arm}" in
    B0)
      policy="${BASE_POLICY}"; plans="${POINTER_PLANS}"; schedule=exact-plan
      ;;
    BC)
      policy="${BASE_POLICY}"; plans="${CANONICAL_PLANS}"; schedule=spad
      ;;
    BS)
      policy="${BS_POLICY}"; plans="${POINTER_PLANS}"; schedule=spad
      extras+=(--spad-backfill)
      ;;
    *) return 2 ;;
  esac

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" --checkpoint-path "${policy}" --prompt-jsonl "${plans}" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" --num-samples 256 \
    --batch-size 8 --temperature 0.7 --seed "${seed}" --preserve-prompt-sample-idx \
    --generation-schedule "${schedule}" "${extras[@]}" \
    >"${RUN}/logs/${arm}_${stream}_body.out" \
    2>"${RUN}/logs/${arm}_${stream}_body.err"
  [[ -f "${cell}/body/_SUCCESS" ]]

  "${PYTHON}" "${SOURCE}/scripts/pad_spad_body_to_planner_denominator.py" \
    --body-rows "${cell}/body/raw_generations.jsonl" \
    --planner-ledger "${PLAN_LEDGER}" --output-dir "${cell}/body_fixed256" \
    --denominator 256

  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body_fixed256" --output-dir "${cell}/raw_generation" \
    --seed "${stream}" --dlm-seed "${seed}" --denominator 256 \
    --source-arm full_axis --arm-label candidate

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${cell}/body/proposal_graphs.pt" --checkpoint "${REFINER}" \
    --output-dir "${cell}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${rseed}" --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${RUN}/logs/${arm}_${stream}_refine.out" \
    2>"${RUN}/logs/${arm}_${stream}_refine.err"
  metrics="${cell}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]]
  result="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
  [[ -f "${result}" ]]

  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${cell}/body_fixed256" --refine-dir "${cell}/refine" \
    --output-dir "${cell}/refined_generation" --arm candidate --repeat 0 \
    --dlm-seed "${seed}" --refiner-seed "${rseed}" --denominator 256
  touch "${cell}/_SUCCESS"
}

index=0
declare -a pids=()
for arm in B0 BC BS; do
  for stream in 17 18; do
    run_cell "${arm}" "${stream}" "${allocated[${index}]}" &
    pids+=("$!")
    index=$((index + 1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status="$?"; done
(( status == 0 ))

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cells = []
for arm in ("B0", "BC", "BS"):
    for stream in (17, 18):
        cell = root / arm / f"stream{stream}"
        body = json.load(open(cell / "body/sample_metrics.json", encoding="utf-8"))
        refine = json.load(open(cell / "refine/refinement_metrics.json", encoding="utf-8"))
        raw = [json.loads(line) for line in open(cell / "raw_generation/generation.jsonl", encoding="utf-8")]
        refined = [json.loads(line) for line in open(cell / "refined_generation/generation.jsonl", encoding="utf-8")]
        assert len(raw) == len(refined) == 256
        assert {row["sample_idx"] for row in raw} == set(range(256))
        assert {row["sample_idx"] for row in refined} == set(range(256))
        cells.append({
            "arm": arm,
            "stream": stream,
            "requested": 256,
            "body_parsed": body["parse_success"],
            "body_plan_match": body["plan_match_success"],
            "graphs": body["graph_success"],
            "refined": refine.get(
                "num_proposals",
                refine.get("num_output_structures", refine.get("output_count")),
            ),
        })
report = {
    "schema": "spad_prospective_generation_v1",
    "cells": cells,
    "model494_tau": 800,
    "one_plan_one_trajectory": True,
    "selection_retry_replacement": False,
}
(root / "GENERATION_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"

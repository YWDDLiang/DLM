#!/usr/bin/env bash
#SBATCH --job-name=spad-paper1000-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=440G
#SBATCH --time=08:00:00
#SBATCH --output=spad-paper1000-gen-%j.out
#SBATCH --error=spad-paper1000-gen-%j.err

set -Eeuo pipefail
IFS=$'\n\t'

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly TRAIN_RUN="${SPAD_BASIN_POSTERIOR_TRAIN_RUN:?export completed K10 training run}"
readonly PLAN_DIR="${SPAD_PLAN_DIR:?export fixed paper-scale Planner cohort}"
readonly AUTH="${SPAD_PAPER1000_AUTHORIZATION_MARKER:?export fixed256 authorization JSON}"
readonly SOURCE_REQUESTED="${SPAD_PAPER_SOURCE_REQUESTED:-1200}"
readonly VALID_TARGET="${SPAD_PAPER_VALID_TARGET:-1000}"
readonly WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"
readonly PLANNER_SEED="${SPAD_PLANNER_SEED:-25}"
readonly STREAM="${SPAD_EVAL_STREAM:-20}"
readonly DLM_SEED="${SPAD_DLM_SEED:-94117}"
readonly REFINER_SEED="${SPAD_REFINER_SEED:-104117}"
readonly SELECTED_TAU="${SPAD_SELECTED_TAU:-800}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON="${SPAD_BASIN_PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
readonly RUN="${ROOT}/runs/spad_basin_paper1000_generation_refine_${SLURM_JOB_ID}"

[[ "${SOURCE_REQUESTED}" =~ ^[0-9]+$ && "${SOURCE_REQUESTED}" -ge "${VALID_TARGET}" ]]
[[ "${VALID_TARGET}" -eq 1000 ]]
[[ "${WORLD_SIZE}" =~ ^[1-4]$ ]]
[[ "${SELECTED_TAU}" -eq 800 ]]

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in \
  "${TRAIN_RUN}/_SUCCESS" \
  "${TRAIN_RUN}/POLICY_PATH" \
  "${TRAIN_RUN}/CAPABILITY_PATH" \
  "${AUTH}" \
  "${PLAN_DIR}/_SUCCESS" \
  "${PLAN_DIR}/manifest.json" \
  "${PLAN_DIR}/plans_for_dlm.jsonl" \
  "${PLAN_DIR}/ledger.jsonl" \
  "${REFINER}" \
  "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
  "${SOURCE}/scripts/select_first_parseable_spad_body.py" \
  "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
  "${SOURCE}/scripts/assemble_grounding_repeat.py"; do
  [[ -f "${path}" ]]
done

"${PYTHON}" - "${AUTH}" <<'PY'
import json
import sys

auth = json.load(open(sys.argv[1], encoding="utf-8"))
assert auth["schema"] == "spad_paper1000_authorization_v1"
assert auth["fixed_denominator"] == 256
assert auth["exact_threshold_counts"] == {"strict": 26, "meta": 128}
assert auth["preregistered_near_miss_counts"] == {"strict": 23, "meta": 125}
assert auth["paper1000_authorized"] is True
assert auth["outcome_based_row_selection"] is False
PY

readonly POLICY="$(<"${TRAIN_RUN}/POLICY_PATH")"
readonly CAPABILITY="$(<"${TRAIN_RUN}/CAPABILITY_PATH")"
[[ -f "${POLICY}/adapter_model.safetensors" && -f "${CAPABILITY}" ]]

"${PYTHON}" - "${PLAN_DIR}" "${SOURCE_REQUESTED}" "${PLANNER_SEED}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
requested = int(sys.argv[2]); planner_seed = int(sys.argv[3])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
ledger = [
    json.loads(line)
    for line in (root / "ledger.jsonl").open(encoding="utf-8")
    if line.strip()
]
plans = [
    json.loads(line)
    for line in (root / "plans_for_dlm.jsonl").open(encoding="utf-8")
    if line.strip()
]
assert manifest["requested"] == requested
assert int(manifest["planner_sampling_seed"]) == planner_seed
assert manifest["outcomes_read"] is False
assert manifest["replacement"] is False
assert manifest["planner_resampled_after_freeze"] is False
assert len(ledger) == requested
assert [int(row["sample_idx"]) for row in ledger] == list(range(requested))
assert {int(row["sample_idx"]) for row in plans} == {
    int(row["sample_idx"]) for row in ledger if row["planner_valid"] is True
}
PY

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq "${WORLD_SIZE}" ]]
gpu_csv="$(IFS=,; printf '%s' "${allocated_gpus[*]}")"

printf '%s\n' \
  $'field\tvalue' \
  $'schema\tspad_basin_paper1000_generation_v1' \
  "source_requested\t${SOURCE_REQUESTED}" \
  "conditional_valid_target\t${VALID_TARGET}" \
  $'selection\tfirst_parseable_CIFs_in_fixed_source_order' \
  $'selection_signal\tparser_only_before_outcomes' \
  $'outcome_rerank_replacement\tfalse' \
  $'launch_gate\texact_26_128_or_preregistered_near_miss_23_125' \
  "planner_seed\t${PLANNER_SEED}" \
  "stream\t${STREAM}" \
  "dlm_seed\t${DLM_SEED}" \
  "refiner_seed\t${REFINER_SEED}" \
  "tau\t${SELECTED_TAU}" \
  >"${RUN}/scientific_contract.tsv"

sample_command=("${PYTHON}")
refine_command=("${PYTHON}")
if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  sample_command+=(
    -m torch.distributed.run --standalone --master_port=37418
    "--nproc_per_node=${WORLD_SIZE}"
  )
  refine_command+=(
    -m torch.distributed.run --standalone --master_port=37419
    "--nproc_per_node=${WORLD_SIZE}"
  )
fi
sample_command+=("${SOURCE}/src/scripts/sample_llada_r5_exact_length.py")
refine_command+=("${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py")

env CUDA_VISIBLE_DEVICES="${gpu_csv}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" \
  "${sample_command[@]}" \
  --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
  --prompt-jsonl "${PLAN_DIR}/plans_for_dlm.jsonl" \
  --output-dir "${RUN}/body_source" --crysllmgen-dir "${CRYS}" \
  --num-samples "${SOURCE_REQUESTED}" --batch-size 8 --temperature 0.7 \
  --seed "${DLM_SEED}" --preserve-prompt-sample-idx \
  --no-repeat-prompt-records --generation-schedule spad \
  --canonicalize-periodic-alias --pbc-min-distance-mask --spad-basin-closure \
  --spad-basin-closure-capability-json "${CAPABILITY}" \
  >"${RUN}/logs/body.out" 2>"${RUN}/logs/body.err"
[[ -f "${RUN}/body_source/_SUCCESS" ]]

"${PYTHON}" "${SOURCE}/scripts/select_first_parseable_spad_body.py" \
  --body-dir "${RUN}/body_source" --planner-ledger "${PLAN_DIR}/ledger.jsonl" \
  --output-dir "${RUN}/body_valid1000" --target "${VALID_TARGET}" \
  >"${RUN}/logs/select.out" 2>"${RUN}/logs/select.err"

"${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
  --body-dir "${RUN}/body_valid1000" --output-dir "${RUN}/raw_generation" \
  --seed "${STREAM}" --dlm-seed "${DLM_SEED}" --denominator "${VALID_TARGET}" \
  --source-arm full_axis --arm-label candidate \
  --method-label SPAD-K10-4104-RAW-PAPER1000 \
  --body-arm-label k10-4104 --schedule-arm-label spad-basin-closure

env CUDA_VISIBLE_DEVICES="${gpu_csv}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${refine_command[@]}" \
  --proposal-graphs "${RUN}/body_valid1000/proposal_graphs.pt" \
  --checkpoint "${REFINER}" --output-dir "${RUN}/refine" \
  --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${REFINER_SEED}" \
  --seed-by-sample-index --diff-steps "${SELECTED_TAU}" \
  --max-proposals "${VALID_TARGET}" \
  >"${RUN}/logs/refine.out" 2>"${RUN}/logs/refine.err"

"${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
  --body-dir "${RUN}/body_valid1000" --refine-dir "${RUN}/refine" \
  --output-dir "${RUN}/refined_generation" --arm candidate --repeat "${STREAM}" \
  --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" \
  --denominator "${VALID_TARGET}" --refinement-steps "${SELECTED_TAU}" \
  --invalid-refined-as-failure \
  --method-label SPAD-K10-4104-MODEL494-PAPER1000 \
  --body-arm-label k10-4104 --schedule-arm-label spad-basin-closure

"${PYTHON}" - "${RUN}" "${PLAN_DIR}" "${SOURCE_REQUESTED}" "${VALID_TARGET}" \
  "${PLANNER_SEED}" "${STREAM}" "${DLM_SEED}" "${REFINER_SEED}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]); plan_dir = pathlib.Path(sys.argv[2]).resolve()
requested, target, planner_seed, stream, dlm_seed, refiner_seed = map(int, sys.argv[3:])
selection = json.loads((root / "body_valid1000/manifest.json").read_text())
body = json.loads((root / "body_source/sample_metrics.json").read_text())
raw_rows = sum(1 for line in (root / "raw_generation/generation.jsonl").open() if line.strip())
refined_rows = sum(
    1 for line in (root / "refined_generation/generation.jsonl").open() if line.strip()
)
assert raw_rows == refined_rows == target
assert selection["selected_parseable_cifs"] == target
assert selection["energy_hull_novelty_sun_read"] is False
report = {
    "schema": "spad_basin_paper1000_generation_final_v1",
    "frozen_plan_dir": str(plan_dir),
    "source_requested": requested,
    "source_decoded": int(body["decoded_samples"]),
    "source_parse_success": int(body["parse_success"]),
    "source_requests_examined": int(selection["source_requests_examined"]),
    "discarded_parser_only_before_target": int(selection["discarded_before_target"]),
    "conditional_valid_denominator": target,
    "raw_rows": raw_rows,
    "refined_rows": refined_rows,
    "planner_seed": planner_seed,
    "stream": stream,
    "dlm_seed": dlm_seed,
    "refiner_seed": refiner_seed,
    "tau": 800,
    "one_plan_one_trajectory": True,
    "parser_only_replenishment_before_outcomes": True,
    "outcome_rerank_replacement": False,
}
(root / "PAPER1000_GENERATION_FINAL.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

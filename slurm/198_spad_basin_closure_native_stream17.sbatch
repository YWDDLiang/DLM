#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-native17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=spad-basin-native17-%j.out
#SBATCH --error=spad-basin-native17-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly TRAIN_RUN="${SPAD_BASIN_CLOSURE_TRAIN_RUN:?export the completed closure-CE run}"
readonly PLAN_DIR="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_native_s17_${SLURM_JOB_ID}"
readonly STREAM=17
readonly DLM_SEED=91117

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in \
  "${TRAIN_RUN}/_SUCCESS" "${TRAIN_RUN}/POLICY_PATH" \
  "${TRAIN_RUN}/CAPABILITY_PATH" "${PLAN_DIR}/_SUCCESS" \
  "${PLAN_DIR}/plans_for_dlm.jsonl" "${PLAN_DIR}/ledger.jsonl" \
  "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py"; do
  [[ -f "${path}" ]]
done
readonly POLICY="$(<"${TRAIN_RUN}/POLICY_PATH")"
readonly CAPABILITY="$(<"${TRAIN_RUN}/CAPABILITY_PATH")"
[[ -f "${POLICY}/adapter_model.safetensors" ]]
[[ -f "${CAPABILITY}" ]]
[[ "$(wc -l <"${PLAN_DIR}/ledger.jsonl")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

readonly CELL="${RUN}/closure_ce/stream17"
mkdir -p "${CELL}"
env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
  --prompt-jsonl "${PLAN_DIR}/plans_for_dlm.jsonl" \
  --output-dir "${CELL}/body" --crysllmgen-dir "${CRYS}" \
  --num-samples 256 --batch-size 8 --temperature 0.7 --seed "${DLM_SEED}" \
  --preserve-prompt-sample-idx --no-repeat-prompt-records \
  --generation-schedule spad --canonicalize-periodic-alias \
  --pbc-min-distance-mask --spad-basin-closure \
  --spad-basin-closure-capability-json "${CAPABILITY}" \
  >"${RUN}/logs/closure_ce_body.out" \
  2>"${RUN}/logs/closure_ce_body.err"
[[ -f "${CELL}/body/_SUCCESS" ]]

"${PYTHON}" "${SOURCE}/scripts/pad_spad_body_to_planner_denominator.py" \
  --body-rows "${CELL}/body/raw_generations.jsonl" \
  --planner-ledger "${PLAN_DIR}/ledger.jsonl" \
  --output-dir "${CELL}/body_fixed256" --denominator 256
"${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
  --body-dir "${CELL}/body_fixed256" --output-dir "${CELL}/raw_generation" \
  --seed "${STREAM}" --dlm-seed "${DLM_SEED}" --denominator 256 \
  --source-arm full_axis --arm-label candidate

"${PYTHON}" - "${RUN}" "${TRAIN_RUN}" "${PLAN_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
train_run = pathlib.Path(sys.argv[2]).resolve(strict=True)
plan_dir = pathlib.Path(sys.argv[3]).resolve(strict=True)
cell = root / "closure_ce" / "stream17"
metrics = json.load((cell / "body" / "sample_metrics.json").open(encoding="utf-8"))
body = [
    json.loads(line)
    for line in (cell / "body" / "raw_generations.jsonl").open(encoding="utf-8")
    if line.strip()
]
attempts = [
    json.loads(line)
    for line in (cell / "raw_generation" / "generation.jsonl").open(encoding="utf-8")
    if line.strip()
]
assert len(attempts) == 256
assert [int(row["ordinal"]) for row in attempts] == list(range(256))
assert all(row.get("spad_basin_closure") is True for row in body)
assert all(
    row["spad_basin_closure_metadata"]["final_geometry_supported"] is True
    for row in body
)
assert all(
    row["spad_basin_closure_metadata"]["stage_order"]
    == ["predictor", "cell", "reverse_species_blocks"]
    for row in body
)
cell_changed = sum(
    int(row["spad_basin_closure_cell_revision_log"].get("changed_components", 0)) > 0
    for row in body
)
blocks = [
    block
    for row in body
    for block in row["spad_basin_closure_species_block_revision_log"]
]
report = {
    "schema": "spad_basin_closure_native_stream17_generation_v1",
    "training_run": str(train_run),
    "plan_dir": str(plan_dir),
    "stream": 17,
    "dlm_seed": 91117,
    "requested": 256,
    "decoded": int(metrics["decoded_samples"]),
    "parsed": int(metrics["parse_success"]),
    "plan_match": int(metrics["plan_match_success"]),
    "graphs": int(metrics["graph_success"]),
    "fixed_attempts": len(attempts),
    "cell_transactions_changed": int(cell_changed),
    "species_blocks_executed": len(blocks),
    "species_blocks_changed": sum(int(block.get("changed_components", 0)) > 0 for block in blocks),
    "species_blocks_restored": sum(bool(block.get("restored_complete_block")) for block in blocks),
    "sites_restored_no_legal_z": sum(int(block.get("restored_site_count", 0)) for block in blocks),
    "schedule": "predictor -> six-token cell closure -> reverse Llama species blocks",
    "one_plan_one_trajectory": True,
    "model494_or_inference_critic": False,
    "selection_retry_replacement": False,
}
(root / "GENERATION_FINAL.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${CELL}/_SUCCESS"
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=spad-post-s18-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=spad-post-s18-gen-%j.out
#SBATCH --error=spad-post-s18-gen-%j.err

set -Eeuo pipefail
IFS=$'\n\t'

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly TRAIN_RUN="${SPAD_BASIN_POSTERIOR_TRAIN_RUN:?export the completed 213 training run}"
readonly SELECTED_TAU="${SPAD_SELECTED_TAU:?export the preregistered stream17-selected tau}"
readonly PRIMARY_ONLY="${SPAD_PRIMARY_ONLY:-false}"
readonly PRIMARY_WORLD_SIZE="${SPAD_PRIMARY_WORLD_SIZE:-4}"
readonly PLAN_DIR="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
readonly CLOSURE_RUN="${ROOT}/runs/spad_basin_closure_ce_39700"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON="${SPAD_BASIN_PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
readonly RUN="${ROOT}/runs/spad_basin_posterior_stream18_generation_refine_${SLURM_JOB_ID}"
readonly STREAM=18
readonly DLM_SEED=92117
readonly REFINER_SEED=102117

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
  "${TRAIN_RUN}/_SUCCESS" \
  "${TRAIN_RUN}/POLICY_PATH" \
  "${TRAIN_RUN}/CAPABILITY_PATH" \
  "${CLOSURE_RUN}/_SUCCESS" \
  "${CLOSURE_RUN}/POLICY_PATH" \
  "${CLOSURE_RUN}/CAPABILITY_PATH" \
  "${PLAN_DIR}/_SUCCESS" \
  "${PLAN_DIR}/plans_for_dlm.jsonl" \
  "${PLAN_DIR}/ledger.jsonl" \
  "${REFINER}" \
  "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  "${SOURCE}/scripts/pad_spad_body_to_planner_denominator.py" \
  "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
  "${SOURCE}/scripts/assemble_grounding_repeat.py"; do
  [[ -f "${path}" ]] || {
    printf 'required input is missing: %s\n' "${path}" >&2
    exit 2
  }
done

"${PYTHON}" - "${TRAIN_RUN}" "${CLOSURE_RUN}" "${PLAN_DIR}" \
  "${RUN}/RESOLVED_INPUTS.json" <<'PY'
import json
import pathlib
import sys

train_run = pathlib.Path(sys.argv[1]).resolve(strict=True)
closure_run = pathlib.Path(sys.argv[2]).resolve(strict=True)
plan_dir = pathlib.Path(sys.argv[3]).resolve(strict=True)
output = pathlib.Path(sys.argv[4])
resolved = {
    "closure_ce": {
        "policy_adapter_path": (closure_run / "POLICY_PATH").read_text(
            encoding="utf-8"
        ).strip(),
        "capability_json": (closure_run / "CAPABILITY_PATH").read_text(
            encoding="utf-8"
        ).strip(),
        "role": "registered_closure_ce_control",
    },
    "k10": {
        "policy_adapter_path": (train_run / "POLICY_PATH").read_text(encoding="utf-8").strip(),
        "capability_json": (train_run / "CAPABILITY_PATH").read_text(encoding="utf-8").strip(),
        "role": "preregistered_primary",
    },
}
for arm, record in resolved.items():
    policy = pathlib.Path(record["policy_adapter_path"]).resolve(strict=True)
    capability = pathlib.Path(record["capability_json"]).resolve(strict=True)
    assert (policy / "adapter_model.safetensors").is_file(), arm
    payload = json.loads(capability.read_text(encoding="utf-8"))
    assert payload["schema"] == "spad_basin_closure_capability_v1", arm
    assert pathlib.Path(payload["checkpoint_path"]).resolve(strict=True) == policy
    record["policy_adapter_path"] = str(policy)
    record["capability_json"] = str(capability)

manifest = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
plans = [
    json.loads(line)
    for line in (plan_dir / "plans_for_dlm.jsonl").open(encoding="utf-8")
    if line.strip()
]
ledger = [
    json.loads(line)
    for line in (plan_dir / "ledger.jsonl").open(encoding="utf-8")
    if line.strip()
]
assert manifest["planner_sampling_seed"] == 23
assert manifest["requested"] == 256
assert manifest["replacement"] is False
assert manifest["planner_resampled_after_freeze"] is False
assert manifest["outcomes_read"] is False
assert len(ledger) == 256
assert [int(row["sample_idx"]) for row in ledger] == list(range(256))
assert len(plans) == int(manifest["planner_valid"])
assert {int(row["sample_idx"]) for row in plans} == {
    int(row["sample_idx"]) for row in ledger if row["planner_valid"] is True
}

output.write_text(
    json.dumps(
        {
            "schema": "spad_basin_posterior_stream18_resolved_inputs_v1",
            "training_run": str(train_run),
            "frozen_plan_dir": str(plan_dir),
            "routes": resolved,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

mapfile -t resolved_paths < <(
  "${PYTHON}" - "${RUN}/RESOLVED_INPUTS.json" <<'PY'
import json
import sys

routes = json.load(open(sys.argv[1], encoding="utf-8"))["routes"]
for arm in ("closure_ce", "k10"):
    print(routes[arm]["policy_adapter_path"])
    print(routes[arm]["capability_json"])
PY
)
[[ "${#resolved_paths[@]}" -eq 4 ]]
readonly CLOSURE_POLICY="${resolved_paths[0]}"
readonly CLOSURE_CAPABILITY="${resolved_paths[1]}"
readonly K10_POLICY="${resolved_paths[2]}"
readonly K10_CAPABILITY="${resolved_paths[3]}"

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
expected_gpus=2
[[ "${PRIMARY_ONLY}" == true ]] && expected_gpus="${PRIMARY_WORLD_SIZE}"
[[ "${#allocated_gpus[@]}" -eq "${expected_gpus}" ]] || {
  printf 'expected %s allocated GPUs, observed %s\n' \
    "${expected_gpus}" "${#allocated_gpus[@]}" >&2
  exit 2
}

printf '%s\n' \
  $'field\tvalue' \
  $'schema\tspad_basin_posterior_stream18_generation_contract_v1' \
  "allocation\t${expected_gpus}_A800" \
  $'stream\t18' \
  $'dlm_seed\t92117' \
  $'refiner_seed\t102117' \
  "selected_tau\t${SELECTED_TAU}" \
  "arms\t$([[ "${PRIMARY_ONLY}" == true ]] && printf k10_primary_only || printf closure_ce_k10_concurrent)" \
  $'primary_arm\tk10' \
  $'automatic_result_choice\tfalse' \
  $'one_plan_one_trajectory\ttrue' \
  $'retry_rerank_replacement\tfalse' \
  >"${RUN}/scientific_contract.tsv"

run_arm() {
  local arm="$1" gpu="$2" world_size="$3" policy capability cell
  local -a sample_command refine_command
  case "${arm}" in
    closure_ce)
      policy="${CLOSURE_POLICY}"
      capability="${CLOSURE_CAPABILITY}"
      ;;
    k10)
      policy="${K10_POLICY}"
      capability="${K10_CAPABILITY}"
      ;;
    *)
      printf 'unknown arm: %s\n' "${arm}" >&2
      return 2
      ;;
  esac
  cell="${RUN}/${arm}/stream18"
  mkdir -p "${cell}"

  sample_command=("${PYTHON}")
  refine_command=("${PYTHON}")
  if [[ "${world_size}" -gt 1 ]]; then
    sample_command+=(
      -m torch.distributed.run --standalone --master_port=37318
      "--nproc_per_node=${world_size}"
    )
    refine_command+=(
      -m torch.distributed.run --standalone --master_port=37319
      "--nproc_per_node=${world_size}"
    )
  fi
  sample_command+=("${SOURCE}/src/scripts/sample_llada_r5_exact_length.py")
  refine_command+=("${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py")

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${SOURCE}/src" \
    "${sample_command[@]}" \
    --model-path "${MODEL}" --checkpoint-path "${policy}" \
    --prompt-jsonl "${PLAN_DIR}/plans_for_dlm.jsonl" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" \
    --num-samples 256 --batch-size 8 --temperature 0.7 --seed "${DLM_SEED}" \
    --preserve-prompt-sample-idx --no-repeat-prompt-records \
    --generation-schedule spad --canonicalize-periodic-alias \
    --pbc-min-distance-mask --spad-basin-closure \
    --spad-basin-closure-capability-json "${capability}" \
    >"${RUN}/logs/${arm}_stream18_body.out" \
    2>"${RUN}/logs/${arm}_stream18_body.err"
  [[ -f "${cell}/body/_SUCCESS" ]]

  "${PYTHON}" "${SOURCE}/scripts/pad_spad_body_to_planner_denominator.py" \
    --body-rows "${cell}/body/raw_generations.jsonl" \
    --planner-ledger "${PLAN_DIR}/ledger.jsonl" \
    --output-dir "${cell}/body_fixed256" --denominator 256

  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body_fixed256" --output-dir "${cell}/raw_generation" \
    --seed "${STREAM}" --dlm-seed "${DLM_SEED}" --denominator 256 \
    --source-arm full_axis --arm-label candidate \
    --method-label "SPAD-BASIN-POSTERIOR-${arm}-RAW" \
    --body-arm-label "${arm}" --schedule-arm-label spad-basin-closure

  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${refine_command[@]}" \
    --proposal-graphs "${cell}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${cell}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${REFINER_SEED}" \
    --seed-by-sample-index --diff-steps "${SELECTED_TAU}" --max-proposals 256 \
    >"${RUN}/logs/${arm}_stream18_refine.out" \
    2>"${RUN}/logs/${arm}_stream18_refine.err"
  [[ -f "${cell}/refine/refinement_metrics.json" ]]

  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${cell}/body_fixed256" --refine-dir "${cell}/refine" \
    --output-dir "${cell}/refined_generation" --arm candidate --repeat 0 \
    --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" \
    --denominator 256 --refinement-steps "${SELECTED_TAU}" \
    --invalid-refined-as-failure \
    --method-label "SPAD-BASIN-POSTERIOR-${arm}-MODEL494" \
    --body-arm-label "${arm}" --schedule-arm-label spad-basin-closure

  [[ "$(wc -l <"${cell}/raw_generation/generation.jsonl")" -eq 256 ]]
  [[ "$(wc -l <"${cell}/refined_generation/generation.jsonl")" -eq 256 ]]
  touch "${cell}/_SUCCESS"
}

if [[ "${PRIMARY_ONLY}" == true ]]; then
  gpu_csv="$(IFS=,; printf '%s' "${allocated_gpus[*]}")"
  run_arm k10 "${gpu_csv}" "${expected_gpus}" &
  CHILD_PIDS+=("$!")
else
  run_arm closure_ce "${allocated_gpus[0]}" 1 &
  CHILD_PIDS+=("$!")
  run_arm k10 "${allocated_gpus[1]}" 1 &
  CHILD_PIDS+=("$!")
fi

arm_failure=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    arm_failure=1
  fi
done
[[ "${arm_failure}" -eq 0 ]] || {
  printf 'one or more fixed stream18 arms failed\n' >&2
  false
}

"${PYTHON}" - "${RUN}" "${PLAN_DIR}" "${SELECTED_TAU}" "${PRIMARY_ONLY}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
plan_dir = pathlib.Path(sys.argv[2]).resolve(strict=True)
tau = int(sys.argv[3])
primary_only = sys.argv[4] == "true"
resolved = json.loads((root / "RESOLVED_INPUTS.json").read_text(encoding="utf-8"))
expected_plans = {
    int(row["sample_idx"]): row["plan_state"]
    for row in (
        json.loads(line)
        for line in (plan_dir / "plans_for_dlm.jsonl").open(encoding="utf-8")
        if line.strip()
    )
}
cells = []
for arm in (("k10",) if primary_only else ("closure_ce", "k10")):
    cell = root / arm / "stream18"
    body_metrics = json.loads(
        (cell / "body" / "sample_metrics.json").read_text(encoding="utf-8")
    )
    body_rows = [
        json.loads(line)
        for line in (cell / "body" / "raw_generations.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    raw_rows = [
        json.loads(line)
        for line in (cell / "raw_generation" / "generation.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    ]
    refined_rows = [
        json.loads(line)
        for line in (cell / "refined_generation" / "generation.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    ]
    assert len(raw_rows) == len(refined_rows) == 256
    assert [int(row["ordinal"]) for row in raw_rows] == list(range(256))
    assert [int(row["ordinal"]) for row in refined_rows] == list(range(256))
    for rows in (raw_rows, refined_rows):
        for row in rows:
            sample_idx = int(row["sample_idx"])
            if sample_idx in expected_plans:
                assert row["plan_state"] == expected_plans[sample_idx]
            else:
                assert row["status"] == "failed"
                assert row["plan_state"] is None
            assert row["retry_or_replacement_used"] is False
    assert all(
        row["diffusion_refinement_applied"] is False
        and row["diffusion_refinement_steps"] == 0
        for row in raw_rows
    )
    assert all(
        row["diffusion_refinement_steps"] in (None, tau) for row in refined_rows
    )
    body_indices = [int(row["sample_idx"]) for row in body_rows]
    assert len(body_indices) == len(set(body_indices))
    assert set(body_indices) <= set(range(256))
    cells.append(
        {
            "arm": arm,
            "role": resolved["routes"][arm]["role"],
            "policy_adapter_path": resolved["routes"][arm]["policy_adapter_path"],
            "capability_json": resolved["routes"][arm]["capability_json"],
            "requested": 256,
            "body_attempts": len(body_rows),
            "body_decoded": int(body_metrics["decoded_samples"]),
            "body_parsed": int(body_metrics["parse_success"]),
            "body_plan_match": int(body_metrics["plan_match_success"]),
            "graphs": int(body_metrics["graph_success"]),
            "raw_succeeded": sum(row["status"] == "succeeded" for row in raw_rows),
            "raw_failed": sum(row["status"] == "failed" for row in raw_rows),
            "refined_succeeded": sum(
                row["status"] == "succeeded" for row in refined_rows
            ),
            "refined_failed": sum(row["status"] == "failed" for row in refined_rows),
        }
    )

report = {
    "schema": "spad_basin_posterior_heldout_stream18_generation_v1",
    "training_run": resolved["training_run"],
    "frozen_plan_dir": str(plan_dir),
    "stream": 18,
    "dlm_seed": 92117,
    "refiner_seed": 102117,
    "selected_tau": tau,
    "selected_tau_source": "completed_stream17_development_calibration",
    "primary_arm": "k10",
    "primary_only": primary_only,
    "cells": cells,
    "one_plan_one_trajectory": True,
    "same_plans_order_and_noise_across_arms": True,
    "all_failures_retained_in_fixed_denominator": True,
    "fixed_denominator": 256,
    "automatic_route_tau_seed_checkpoint_choice": False,
    "retry_rerank_replacement": False,
}
(root / "GENERATION_FINAL.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, sort_keys=True))
PY

trap - ERR INT TERM
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

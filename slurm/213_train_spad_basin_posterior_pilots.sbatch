#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-post
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=spad-basin-post-%j.out
#SBATCH --error=spad-basin-post-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly LABEL_RUN="${SPAD_BASIN_LABEL_RUN:?export the completed labelled preflight run}"
readonly LABELLED_GROUPS="${SPAD_BASIN_LABELLED_GROUPS:-${LABEL_RUN}/labelled_groups.jsonl}"
readonly AUTHORIZATION="${LABEL_RUN}/PRELIGHT_TRAINING_AUTHORIZED"
readonly MODEL="${SPAD_BASIN_MODEL_PATH:-/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct}"
readonly CLOSURE_POLICY="${SPAD_BASIN_CLOSURE_CHECKPOINT:-${ROOT}/runs/spad_basin_closure_ce_39700/train/checkpoints/step-1696}"
readonly SFT_DATA="${ROOT}/data/spad_basin_closure_sft_v1_20260904"
readonly TRAINER="${SOURCE}/src/scripts/train_spad_basin_posterior.py"
readonly PYTHON="${SPAD_BASIN_PYTHON:-/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python}"
readonly RUN="${ROOT}/runs/spad_basin_posterior_pilots_${SLURM_JOB_ID}"
readonly E0_MASTER_PORT="${SPAD_BASIN_E0_MASTER_PORT:-37117}"
readonly K10_MASTER_PORT="${SPAD_BASIN_K10_MASTER_PORT:-37118}"

for path in \
  "${LABELLED_GROUPS}" \
  "${AUTHORIZATION}" \
  "${CLOSURE_POLICY}/adapter_model.safetensors" \
  "${CLOSURE_POLICY}/spad_basin_closure_capability.json" \
  "${SFT_DATA}/train.jsonl" \
  "${SFT_DATA}/_SUCCESS" \
  "${TRAINER}"; do
  [[ -f "${path}" ]] || {
    printf 'required input is missing: %s\n' "${path}" >&2
    exit 2
  }
done

[[ "${E0_MASTER_PORT}" != "${K10_MASTER_PORT}" ]] || {
  printf 'the two routes require separate master ports\n' >&2
  exit 2
}

"${PYTHON}" - "${LABELLED_GROUPS}" "${AUTHORIZATION}" <<'PY'
import json
import pathlib
import sys

groups = pathlib.Path(sys.argv[1])
authorization = pathlib.Path(sys.argv[2])
rows = []
with groups.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            value = json.loads(line)
            assert isinstance(value, dict)
            rows.append(value)
assert len(rows) == 128, len(rows)
assert [int(row["state"]["sample_idx"]) for row in rows] == list(range(128))
text = authorization.read_text(encoding="utf-8").strip()
if text.startswith("{"):
    report = json.loads(text)
    assert report.get("authorized") is not False
    assert report.get("passed") is not False
PY

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]] || {
  printf 'expected exactly four allocated GPUs, observed %s\n' "${#allocated_gpus[@]}" >&2
  exit 2
}
readonly E0_GPUS="${allocated_gpus[0]},${allocated_gpus[1]}"
readonly K10_GPUS="${allocated_gpus[2]},${allocated_gpus[3]}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
printf '%s\n' \
  $'field\tvalue' \
  $'schema\tspad_basin_posterior_pilot_wrapper_v1' \
  $'allocation\t4_A800_16_CPU' \
  $'routes\te0_and_k10_concurrent' \
  $'primary_route\tk10_basin_consistent' \
  $'diagnostic_route\te0_single_point' \
  $'groups\t128_ordered_common' \
  $'posterior_passes\t4' \
  $'optimizer_updates_per_route\t512' \
  $'objective_schedule\tclean_CE_then_posterior' \
  $'automatic_result_choice\tfalse' \
  >"${RUN}/scientific_contract.tsv"

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

launch_route() {
  local route="$1"
  local value_field="$2"
  local seed="$3"
  local master_port="$4"
  local devices="$5"

  CUDA_VISIBLE_DEVICES="${devices}" \
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" -m torch.distributed.run \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --master-port "${master_port}" \
      --nproc_per_node=2 \
      "${TRAINER}" \
      --model-path "${MODEL}" \
      --checkpoint-path "${CLOSURE_POLICY}" \
      --labelled-groups "${LABELLED_GROUPS}" \
      --sft-data-dir "${SFT_DATA}" \
      --output-dir "${RUN}/${route}" \
      --authorization-marker "${AUTHORIZATION}" \
      --route-name "${route}" \
      --value-field "${value_field}" \
      --seed "${seed}" \
      >"${RUN}/logs/${route}.out" 2>"${RUN}/logs/${route}.err" &
  CHILD_PIDS+=("$!")
}

launch_route \
  e0 \
  terminal_single_point_energy_eV_per_atom \
  99017 \
  "${E0_MASTER_PORT}" \
  "${E0_GPUS}"
launch_route \
  k10 \
  terminal_relax_k10_energy_eV_per_atom \
  99018 \
  "${K10_MASTER_PORT}" \
  "${K10_GPUS}"

route_failure=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    route_failure=1
    cleanup_children
  fi
done
[[ "${route_failure}" -eq 0 ]] || {
  printf 'one or both fixed pilot routes failed\n' >&2
  false
}

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "e0": ("terminal_single_point_energy_eV_per_atom", 99017),
    "k10": ("terminal_relax_k10_energy_eV_per_atom", 99018),
}
summary = {
    "schema": "spad_basin_posterior_pilot_routes_v1",
    "automatic_result_choice": False,
    "primary_route": "k10",
    "diagnostic_route": "e0",
    "routes": {},
}
for route, (field, seed) in expected.items():
    directory = root / route
    assert (directory / "_SUCCESS").is_file()
    report = json.load((directory / "TRAIN_FINAL.json").open(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["route_name"] == route
    assert report["value_field"] == field
    assert report["seed"] == seed
    assert report["optimizer_updates"] == 512
    assert report["objective_counts"] == {
        "clean_ce": 256,
        "transaction_posterior": 256,
    }
    assert report["each_group_seen_exactly"] == 4
    assert report["same_step_scalar_mixture"] is False
    assert report["checkpoint"]["only_final_policy_saved"] is True
    summary["routes"][route] = {
        "run": str(directory),
        "value_field": field,
        "seed": seed,
        "policy_adapter_path": report["checkpoint"]["policy_adapter_path"],
        "capability_json": report["checkpoint"]["capability_json"],
    }
(root / "ROUTES.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

trap - ERR INT TERM
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

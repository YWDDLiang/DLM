#!/usr/bin/env bash
#SBATCH --job-name=full-mp20-value
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=full-mp20-value-%j.out
#SBATCH --error=full-mp20-value-%j.err

set -Eeuo pipefail

# Submit this wrapper once per route.  The orchestrator may run the two jobs in
# parallel, but this file deliberately launches exactly one two-GPU route.
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly LABELLED_GROUPS="${FULL_MP20_LABELLED_GROUPS:?}"
readonly ROUTE="${FULL_MP20_VALUE_ROUTE:?single_point_full or basin_consistent_full}"
readonly AUDIT_ONLY="${FULL_MP20_GRADIENT_AUDIT_ONLY:-0}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly SFT_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly PHASE="$([[ "${AUDIT_ONLY}" == 1 ]] && printf audit || printf train)"
readonly RUN="${ROOT}/runs/full_mp20_transaction_value_${ROUTE}_${PHASE}_${SLURM_JOB_ID}"

case "${ROUTE}" in
  single_point_full|basin_consistent_full) ;;
  *) printf 'invalid FULL_MP20_VALUE_ROUTE: %s\n' "${ROUTE}" >&2; exit 2 ;;
esac
case "${AUDIT_ONLY}" in
  0|1) ;;
  *) printf 'FULL_MP20_GRADIENT_AUDIT_ONLY must be 0 or 1\n' >&2; exit 2 ;;
esac

extra_args=()
if [[ "${AUDIT_ONLY}" == 1 ]]; then
  extra_args+=(--audit-only)
else
  readonly POSTERIOR_SCALE="${FULL_MP20_POSTERIOR_GRADIENT_SCALE:?run both audits and provide their one shared preregistered scale}"
  case "${POSTERIOR_SCALE}" in
    0.25|0.5|1|1.0|2|2.0|4|4.0) ;;
    *) printf 'posterior scale is outside the preregistered set\n' >&2; exit 2 ;;
  esac
  extra_args+=(--posterior-gradient-scale "${POSTERIOR_SCALE}")
fi

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

[[ -f "${LABELLED_GROUPS}" ]]
[[ -f "${SFT_DATA}/_SUCCESS" ]]
[[ -f "${SFT_DATA}/train.jsonl" ]]
[[ -f "${BS_POLICY}/adapter_model.safetensors" ]]
"${PYTHON}" - "${LABELLED_GROUPS}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = 0
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        value = json.loads(line)
        assert isinstance(value, dict)
        rows += 1
assert rows == 27136, rows
PY

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated_gpus[@]}" -eq 2 ]]

env \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${SOURCE}/src/scripts/train_full_mp20_transaction_value.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BS_POLICY}" \
  --labelled-groups "${LABELLED_GROUPS}" \
  --sft-data-dir "${SFT_DATA}" \
  --output-dir "${RUN}/train" \
  --route "${ROUTE}" \
  --seed 99017 \
  --require-llama-program \
  "${extra_args[@]}" \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

"${PYTHON}" - "${RUN}" "${ROUTE}" "${AUDIT_ONLY}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
route = sys.argv[2]
audit_only = sys.argv[3] == "1"
cell = root / "train"
assert (cell / "_SUCCESS").is_file()
if audit_only:
    audit = json.load(open(cell / "AUDIT_FINAL.json"))
    assert audit["schema"] == "full_mp20_initial_gradient_audit_v1"
    assert audit["route"] == route and audit["optimizer_updates"] == 0
    assert audit["audit_batches"] == 5 and audit["passed"] is True
    assert audit["finite_candidate_kl_budget_nats"] == 0.05
    assert audit["same_batch_half_half_scalar_mixture"] is False
    assert audit["per_batch_inverse_scaling"] is False
    (root / "AUDIT_FINAL.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0)
final = json.load(open(cell / "TRAIN_FINAL.json"))
assert final["schema"] == "full_mp20_transaction_value_train_v1"
assert final["status"] == "success" and final["route"] == route
assert final["world_size"] == 2
assert final["local_microexamples"] == 8
assert final["global_batch_size"] == 16
assert final["optimizer_updates"] == 3392
assert final["objective_counts"] == {
    "clean_ce": 1696,
    "transaction_posterior": 1696,
}
assert final["clean_source_coverage"] == 27136
assert final["posterior_source_coverage"] == 27136
assert final["manual_gradient_allreduce_mean"] is True
assert final["finite_candidate_kl_budget_nats"] == 0.05
assert final["same_batch_half_half_scalar_mixture"] is False
assert final["initial_gradient_audit"]["audit_batches"] == 5
assert final["gradient_scale_contract"]["frozen_for_all_updates"] is True
assert final["gradient_scale_contract"]["per_batch_inverse_scaling"] is False
assert final["checkpoint"]["only_step3392"] is True
assert [p.name for p in (cell / "checkpoints").iterdir() if p.is_dir()] == ["step-3392"]
(root / "TRAIN_FINAL.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

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
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly SFT_DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/full_mp20_transaction_value_${ROUTE}_${SLURM_JOB_ID}"

case "${ROUTE}" in
  single_point_full|basin_consistent_full) ;;
  *) printf 'invalid FULL_MP20_VALUE_ROUTE: %s\n' "${ROUTE}" >&2; exit 2 ;;
esac

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
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

"${PYTHON}" - "${RUN}" "${ROUTE}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
route = sys.argv[2]
cell = root / "train"
assert (cell / "_SUCCESS").is_file()
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
assert final["checkpoint"]["only_step3392"] is True
assert [p.name for p in (cell / "checkpoints").iterdir() if p.is_dir()] == ["step-3392"]
(root / "TRAIN_FINAL.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

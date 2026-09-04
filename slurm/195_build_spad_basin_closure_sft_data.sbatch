#!/usr/bin/env bash
#SBATCH --job-name=spad-closure-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=spad-closure-data-%j.out
#SBATCH --error=spad-closure-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT to the deployed closure branch}"
readonly TEACHER="${ROOT}/data/c3fd_native_teacher_sft_canonical_v2_20260902"
readonly POINTER="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly OUTPUT="${ROOT}/data/spad_basin_closure_sft_v1_20260904"
readonly RUN="${ROOT}/runs/build_spad_basin_closure_sft_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

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

[[ ! -e "${OUTPUT}" ]]
for path in \
  "${SOURCE}/scripts/build_spad_basin_closure_sft_data.py" \
  "${TEACHER}/train.jsonl" "${TEACHER}/val.jsonl" \
  "${TEACHER}/vocab_tokens.txt" \
  "${POINTER}/train.jsonl" "${POINTER}/val.jsonl"; do
  [[ -f "${path}" ]]
done

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/build_spad_basin_closure_sft_data.py" \
  --teacher-data-dir "${TEACHER}" \
  --pointer-data-dir "${POINTER}" \
  --output-dir "${OUTPUT}" \
  --seed 20260904 \
  >"${RUN}/logs/build.out" 2>"${RUN}/logs/build.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l <"${OUTPUT}/val.jsonl")" -eq 9047 ]]

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.load((root / "manifest.json").open(encoding="utf-8"))
assert manifest["schema"] == "spad_basin_closure_sft_manifest_v1"
assert manifest["schedule"] == "cell_then_reverse_llama_species_blocks_v1"
assert manifest["splits"]["train"]["rows"] == 27136
assert manifest["splits"]["val"]["rows"] == 9047
assert manifest["exact_7_plus_4N"] is True
assert manifest["exact_N_elements_never_masked"] is True
assert manifest["outcomes_read"] is False
expected_classes = {
    "cell_sequential_component",
    "reverse_species_block_component",
}
for split, expected_rows in (("train", 27136), ("val", 9047)):
    classes = manifest["splits"][split]["mask_classes"]
    assert set(classes) == expected_classes
    assert all(int(classes[name]) > 0 for name in expected_classes)
    seen = 0
    with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["schema"] == "rollout_matched_transition_v1"
            assert row["closure_schema"] == "spad_basin_closure_sft_v1"
            assert row["closure"]["schedule"] == manifest["schedule"]
            assert row["outcomes_read"] is False
            assert row["answer"] == row["source_answer"]
            forced = {int(value) for value in row["forced_mask_positions"]}
            loss = {int(value) for value in row["loss_positions"]}
            num_atoms = int(row["plan_state"]["N"])
            protected = {0, *(7 + 4 * slot for slot in range(num_atoms))}
            assert loss and loss <= forced
            assert not forced & protected
            assert len(row["species_program"]) == len(row["plan_state"]["elements"])
            assert set(row["species_program"]) == set(row["plan_state"]["elements"])
            seen += 1
    assert seen == expected_rows
PY

printf '%s\n' "${OUTPUT}" >"${RUN}/DATA_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

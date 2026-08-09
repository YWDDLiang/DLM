#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_RUN="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
REPAIR_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_assembly_256_path_repair_v11"
AUDIT64_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw64_smact4_audit_input_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SELF_SHA256="${1:?expected preparation-script SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1
PARENT_EXECUTION="${PARENT_RUN}/source/${EXECUTION_REL}"
EXPECTED_PARENT_SOURCE_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_PARENT_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_AUDIT64_MANIFEST_SHA256=21e813fe3c351cde900dfda51506892f616a375904a1beb6d30241e1a32b33e3
OLD_RUN_NAME=20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7
PARENT_RUN_NAME=20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test ! -e "${REPAIR_ROOT}"
test "$(sha256sum "${PARENT_RUN}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_SOURCE_SHA256}"
test "$(sha256sum "${PARENT_RUN}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_ARCHIVE_SHA256}"
(cd "${PARENT_RUN}/source" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "${AUDIT64_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUDIT64_MANIFEST_SHA256}"
test -f "${AUDIT64_ROOT}/_SUCCESS"
for arm in p0 sft_v2 sft_v2_c; do
  test "$(wc -l < "${PARENT_RUN}/planner64/${arm}/raw_generations.jsonl")" -eq 64
  test -f "${PARENT_RUN}/status/planner64_${arm}_SUCCESS"
done
test ! -e "${PARENT_RUN}/assembly64_submission_record.json"
test ! -e "${PARENT_RUN}/planner64/terminal"
test ! -e "${PARENT_RUN}/planner64/_SUCCESS"
test ! -e "${PARENT_RUN}/planner64/_SCIENTIFIC_STOP"
test ! -e "${PARENT_RUN}/planner64/_FAILED"

mkdir -p "${REPAIR_ROOT}/launchers" "${REPAIR_ROOT}/status"
cp "${SELF}" "${REPAIR_ROOT}/prepare_v11_assembly_256_path_repair_on_a800.sh"

export PARENT_RUN PARENT_EXECUTION REPAIR_ROOT OLD_RUN_NAME PARENT_RUN_NAME
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

parent = Path(os.environ["PARENT_EXECUTION"])
root = Path(os.environ["REPAIR_ROOT"])
out = root / "launchers"
old = os.environ["OLD_RUN_NAME"]
current = os.environ["PARENT_RUN_NAME"]

mapping = {
    "submit_assemble64_once.sh": "submit_assemble64_v11_once.sh",
    "assemble64.sbatch": "assemble64_v11.sbatch",
    "assemble_stage.sbatch": "assemble_stage_v11.sbatch",
    "submit_256_once.sh": "submit_256_v11_once.sh",
    "planner256.sbatch": "planner256_v11.sbatch",
    "submit_assemble256_once.sh": "submit_assemble256_v11_once.sh",
    "assemble256.sbatch": "assemble256_v11.sbatch",
}
execution_line = (
    'EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
    'execution/h1_chemistry_first_sft_v2_v1"\n'
)
source_line = 'SOURCE_ROOT="${RUN_ROOT}/source"\n'
repair_line = f'REPAIR_ROOT="{root}"\n'

for source_name, output_name in mapping.items():
    source = parent / source_name
    text = source.read_text(encoding="utf-8")
    if "\r" in text:
        raise SystemExit(f"CR byte in parent launcher: {source_name}")
    count = text.count(old)
    if count < 1:
        raise SystemExit(f"missing frozen V7 path in {source_name}")
    text = text.replace(old, current)

    if source_name in {
        "submit_assemble64_once.sh",
        "submit_256_once.sh",
        "submit_assemble256_once.sh",
    }:
        if text.count(execution_line) != 1:
            raise SystemExit(f"execution-line identity mismatch: {source_name}")
        text = text.replace(execution_line, execution_line + repair_line, 1)
    if source_name in {"assemble64.sbatch", "assemble256.sbatch"}:
        if text.count(source_line) != 1:
            raise SystemExit(f"source-line identity mismatch: {source_name}")
        text = text.replace(source_line, source_line + repair_line, 1)

    substitutions = {
        "submit_assemble64_once.sh": (
            '"${EXECUTION_DIR}/assemble64.sbatch"',
            '"${REPAIR_ROOT}/launchers/assemble64_v11.sbatch"',
        ),
        "assemble64.sbatch": (
            'exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"',
            'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v11.sbatch"',
        ),
        "submit_256_once.sh": (
            '"${EXECUTION_DIR}/planner256.sbatch"',
            '"${REPAIR_ROOT}/launchers/planner256_v11.sbatch"',
        ),
        "submit_assemble256_once.sh": (
            '"${EXECUTION_DIR}/assemble256.sbatch"',
            '"${REPAIR_ROOT}/launchers/assemble256_v11.sbatch"',
        ),
        "assemble256.sbatch": (
            'exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"',
            'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v11.sbatch"',
        ),
    }
    if source_name in substitutions:
        before, after = substitutions[source_name]
        if text.count(before) != 1:
            raise SystemExit(f"launcher redirect identity mismatch: {source_name}")
        text = text.replace(before, after, 1)

    if old in text:
        raise SystemExit(f"stale V7 path remains: {source_name}")
    (out / output_name).write_text(text, encoding="utf-8", newline="\n")
PY

for path in "${REPAIR_ROOT}"/launchers/*; do
  bash -n "${path}"
  grep -Fq "${PARENT_RUN_NAME}" "${path}"
  if grep -Fq "${OLD_RUN_NAME}" "${path}"; then
    echo "stale V7 path in ${path}" >&2
    exit 3
  fi
done

export EXPECTED_PARENT_SOURCE_SHA256 EXPECTED_PARENT_ARCHIVE_SHA256
export EXPECTED_AUDIT64_MANIFEST_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["REPAIR_ROOT"])
launchers = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((root / "launchers").iterdir())
    if path.is_file()
}
payload = {
    "schema": "h1_chemistry_first_assembly_256_path_repair_v11",
    "status": "pass",
    "repair_root": str(root),
    "parent_run_root": os.environ["PARENT_RUN"],
    "parent_source_inventory_sha256": os.environ["EXPECTED_PARENT_SOURCE_SHA256"],
    "parent_source_archive_sha256": os.environ["EXPECTED_PARENT_ARCHIVE_SHA256"],
    "raw64_local_smact4_manifest_sha256": os.environ["EXPECTED_AUDIT64_MANIFEST_SHA256"],
    "repair_scope": "launcher_run_root_and_launcher_redirect_only",
    "launchers": launchers,
    "raw64_bytes_changed": False,
    "training_or_checkpoint_changed": False,
    "model_data_prompt_seed_optimizer_scheduler_ledger_evaluator_gate_changed": False,
    "smact4_executed_on_a800": False,
    "broad_tests_repeated": False,
}
(root / "V11_REPAIR_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

(
  cd "${REPAIR_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum prepare_v11_assembly_256_path_repair_on_a800.sh V11_REPAIR_RECORD.json
) > "${REPAIR_ROOT}/V11_SHA256.txt"
(cd "${REPAIR_ROOT}" && sha256sum -c V11_SHA256.txt)
touch "${REPAIR_ROOT}/status/preparation_SUCCESS"
find "${REPAIR_ROOT}" -type f -exec chmod 400 {} +
find "${REPAIR_ROOT}" -type d -exec chmod 500 {} +
cat "${REPAIR_ROOT}/V11_REPAIR_RECORD.json"

#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_RUN="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
FAILED_REPAIR_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_assembly_256_path_repair_v11"
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_slurm_list_serialization_repair_v12"
AUDIT64_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw64_smact4_audit_input_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SELF_SHA256="${1:?expected preparation-script SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1
PARENT_EXECUTION="${PARENT_RUN}/source/${EXECUTION_REL}"
EXPECTED_PARENT_SOURCE_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_PARENT_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_AUDIT64_MANIFEST_SHA256=21e813fe3c351cde900dfda51506892f616a375904a1beb6d30241e1a32b33e3
EXPECTED_FAILED_ASSEMBLY_JOB_ID=31229
OLD_RUN_NAME=20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7
RUN_NAME=20260809_h1_chemistry_first_sft_v2_v10_slurm_list_serialization_repair_v12

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test ! -e "${RUN_ROOT}"
test "$(sha256sum "${PARENT_RUN}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_SOURCE_SHA256}"
test "$(sha256sum "${PARENT_RUN}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_ARCHIVE_SHA256}"
(cd "${PARENT_RUN}/source" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "${AUDIT64_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUDIT64_MANIFEST_SHA256}"
test -f "${AUDIT64_ROOT}/_SUCCESS"
test -f "${FAILED_REPAIR_ROOT}/status/preparation_SUCCESS"
test -f "${PARENT_RUN}/planner64/_FAILED"
test "$(tr -d '[:space:]' < "${PARENT_RUN}/status/submitted_assemble64_job_id.txt")" = \
  "${EXPECTED_FAILED_ASSEMBLY_JOB_ID}"
test "$(tr -d '[:space:]' < "${PARENT_RUN}/status/assemble64_exit_code.txt")" = 2
grep -Fq 'local exact-SMACT4 stage manifest identity mismatch' \
  "${PARENT_RUN}/logs/${EXPECTED_FAILED_ASSEMBLY_JOB_ID}_assemble64.err"
for arm in p0 sft_v2 sft_v2_c; do
  test "$(wc -l < "${PARENT_RUN}/planner64/${arm}/raw_generations.jsonl")" -eq 64
  test -f "${PARENT_RUN}/status/planner64_${arm}_SUCCESS"
done
for candidate in sft_v2 sft_v2_c; do
  test -f "${PARENT_RUN}/status/train_${candidate}_SUCCESS"
  test -f "${PARENT_RUN}/training/${candidate}/terminal_report.json"
done

mkdir -p \
  "${RUN_ROOT}/launchers" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/planner64" \
  "${RUN_ROOT}/preflight" \
  "${RUN_ROOT}/status"
cp "${SELF}" "${RUN_ROOT}/prepare_v12_slurm_list_serialization_repair_on_a800.sh"
ln -s "${PARENT_RUN}/source" "${RUN_ROOT}/source"
ln -s "${PARENT_RUN}/source_archive.tar.gz" "${RUN_ROOT}/source_archive.tar.gz"
ln -s "${PARENT_RUN}/training" "${RUN_ROOT}/training"
for arm in p0 sft_v2 sft_v2_c; do
  ln -s "${PARENT_RUN}/planner64/${arm}" "${RUN_ROOT}/planner64/${arm}"
done
cp "${PARENT_RUN}/submission_record.json" "${RUN_ROOT}/submission_record.json"
sha256sum "${RUN_ROOT}/submission_record.json" > "${RUN_ROOT}/submission_record.sha256"
for name in \
  data_SUCCESS \
  planner64_p0_exit_code.txt planner64_p0_SUCCESS \
  planner64_sft_v2_exit_code.txt planner64_sft_v2_SUCCESS \
  planner64_sft_v2_c_exit_code.txt planner64_sft_v2_c_SUCCESS \
  submitted_planner64_job_id.txt submitted_train_job_id.txt \
  train_sft_v2_exit_code.txt train_sft_v2_SUCCESS \
  train_sft_v2_c_exit_code.txt train_sft_v2_c_SUCCESS; do
  cp "${PARENT_RUN}/status/${name}" "${RUN_ROOT}/status/${name}"
done

export PARENT_RUN PARENT_EXECUTION FAILED_REPAIR_ROOT RUN_ROOT AUDIT64_ROOT
export OLD_RUN_NAME RUN_NAME EXPECTED_FAILED_ASSEMBLY_JOB_ID
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

parent = Path(os.environ["PARENT_EXECUTION"])
root = Path(os.environ["RUN_ROOT"])
out = root / "launchers"
old = os.environ["OLD_RUN_NAME"]
current = os.environ["RUN_NAME"]

mapping = {
    "submit_assemble64_once.sh": "submit_assemble64_v12_once.sh",
    "assemble64.sbatch": "assemble64_v12.sbatch",
    "assemble_stage.sbatch": "assemble_stage_v12.sbatch",
    "submit_256_once.sh": "submit_256_v12_once.sh",
    "planner256.sbatch": "planner256_v12.sbatch",
    "submit_assemble256_once.sh": "submit_assemble256_v12_once.sh",
    "assemble256.sbatch": "assemble256_v12.sbatch",
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
    if text.count(old) < 1:
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

    redirects = {
        "submit_assemble64_once.sh": (
            '"${EXECUTION_DIR}/assemble64.sbatch"',
            '"${REPAIR_ROOT}/launchers/assemble64_v12.sbatch"',
        ),
        "assemble64.sbatch": (
            'exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"',
            'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v12.sbatch"',
        ),
        "submit_256_once.sh": (
            '"${EXECUTION_DIR}/planner256.sbatch"',
            '"${REPAIR_ROOT}/launchers/planner256_v12.sbatch"',
        ),
        "submit_assemble256_once.sh": (
            '"${EXECUTION_DIR}/assemble256.sbatch"',
            '"${REPAIR_ROOT}/launchers/assemble256_v12.sbatch"',
        ),
        "assemble256.sbatch": (
            'exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"',
            'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v12.sbatch"',
        ),
    }
    if source_name in redirects:
        before, after = redirects[source_name]
        if text.count(before) != 1:
            raise SystemExit(f"launcher redirect identity mismatch: {source_name}")
        text = text.replace(before, after, 1)

    if source_name == "submit_assemble64_once.sh":
        marker = '\ncommon_export="'
        if text.count(marker) != 1:
            raise SystemExit("raw64 common-export identity mismatch")
        text = text.replace(
            marker,
            '\nAUDITED_ARMS_SLURM="${AUDITED_ARMS//,/:}"\ncommon_export="',
            1,
        )
        old_export = ',AUDITED_ARMS=${AUDITED_ARMS}"'
        new_export = ',AUDITED_ARMS_SLURM=${AUDITED_ARMS_SLURM}"'
        if text.count(old_export) != 1:
            raise SystemExit("raw64 audited-arm export identity mismatch")
        text = text.replace(old_export, new_export, 1)
    elif source_name == "assemble64.sbatch":
        exec_line = 'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v12.sbatch"'
        prelude = (
            ': "${AUDITED_ARMS_SLURM:?missing serialized exact-audit arms}"\n'
            'export AUDITED_ARMS="${AUDITED_ARMS_SLURM//:/,}"\n'
        )
        if text.count(exec_line) != 1:
            raise SystemExit("raw64 wrapper-exec identity mismatch")
        text = text.replace(exec_line, prelude + exec_line, 1)
    elif source_name == "submit_256_once.sh":
        unused_export = ',EXPECTED_CANDIDATES=${EXPECTED_CANDIDATES}"'
        if text.count(unused_export) != 1:
            raise SystemExit("raw256 unused candidate export identity mismatch")
        text = text.replace(unused_export, '"', 1)
    elif source_name == "submit_assemble256_once.sh":
        marker = '\ncommon_export="'
        if text.count(marker) != 1:
            raise SystemExit("raw256 assembly common-export identity mismatch")
        text = text.replace(
            marker,
            '\nAUDITED_ARMS_SLURM="${AUDITED_ARMS//,/:}"\n'
            'EXPECTED_CANDIDATES_SLURM="${EXPECTED_CANDIDATES//,/:}"\n'
            'common_export="',
            1,
        )
        replacements = {
            ',EXPECTED_CANDIDATES=${EXPECTED_CANDIDATES},':
                ',EXPECTED_CANDIDATES_SLURM=${EXPECTED_CANDIDATES_SLURM},',
            ',AUDITED_ARMS=${AUDITED_ARMS}"':
                ',AUDITED_ARMS_SLURM=${AUDITED_ARMS_SLURM}"',
        }
        for before, after in replacements.items():
            if text.count(before) != 1:
                raise SystemExit(f"raw256 list export identity mismatch: {before}")
            text = text.replace(before, after, 1)
    elif source_name == "assemble256.sbatch":
        old_check = ': "${EXPECTED_CANDIDATES:?missing passing raw64 candidates}"\n'
        new_check = (
            ': "${EXPECTED_CANDIDATES_SLURM:?missing serialized passing candidates}"\n'
            ': "${AUDITED_ARMS_SLURM:?missing serialized exact-audit arms}"\n'
            'export EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES_SLURM//:/,}"\n'
            'export AUDITED_ARMS="${AUDITED_ARMS_SLURM//:/,}"\n'
        )
        if text.count(old_check) != 1:
            raise SystemExit("raw256 wrapper candidate check identity mismatch")
        text = text.replace(old_check, new_check, 1)

    if old in text:
        raise SystemExit(f"stale V7 path remains: {source_name}")
    (out / output_name).write_text(text, encoding="utf-8", newline="\n")
PY

for path in "${RUN_ROOT}"/launchers/*; do
  bash -n "${path}"
  grep -Fq "${RUN_NAME}" "${path}"
  if grep -Fq "${OLD_RUN_NAME}" "${path}"; then
    echo "stale V7 path in ${path}" >&2
    exit 3
  fi
done
grep -Fq 'AUDITED_ARMS_SLURM="${AUDITED_ARMS//,/:}"' \
  "${RUN_ROOT}/launchers/submit_assemble64_v12_once.sh"
grep -Fq 'export AUDITED_ARMS="${AUDITED_ARMS_SLURM//:/,}"' \
  "${RUN_ROOT}/launchers/assemble64_v12.sbatch"
grep -Fq 'EXPECTED_CANDIDATES_SLURM="${EXPECTED_CANDIDATES//,/:}"' \
  "${RUN_ROOT}/launchers/submit_assemble256_v12_once.sh"
if grep -Fq 'AUDITED_ARMS=${AUDITED_ARMS}' \
  "${RUN_ROOT}/launchers/submit_assemble64_v12_once.sh" \
  "${RUN_ROOT}/launchers/submit_assemble256_v12_once.sh"; then
  echo 'unserialized audited-arm export remains' >&2
  exit 3
fi

export EXPECTED_PARENT_SOURCE_SHA256 EXPECTED_PARENT_ARCHIVE_SHA256
export EXPECTED_AUDIT64_MANIFEST_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
root = Path(os.environ["RUN_ROOT"])
parent = Path(os.environ["PARENT_RUN"])
launchers = {
    path.name: sha(path)
    for path in sorted((root / "launchers").iterdir())
    if path.is_file()
}
raw64 = {
    arm: sha(parent / "planner64" / arm / "raw_generations.jsonl")
    for arm in ("p0", "sft_v2", "sft_v2_c")
}
training_terminals = {
    candidate: sha(parent / "training" / candidate / "terminal_report.json")
    for candidate in ("sft_v2", "sft_v2_c")
}
payload = {
    "schema": "h1_chemistry_first_slurm_list_serialization_repair_v12",
    "status": "pass",
    "run_root": str(root),
    "parent_run_root": str(parent),
    "parent_source_inventory_sha256": os.environ["EXPECTED_PARENT_SOURCE_SHA256"],
    "parent_source_archive_sha256": os.environ["EXPECTED_PARENT_ARCHIVE_SHA256"],
    "raw64_local_smact4_manifest_sha256": os.environ["EXPECTED_AUDIT64_MANIFEST_SHA256"],
    "failed_assembly_job_id": os.environ["EXPECTED_FAILED_ASSEMBLY_JOB_ID"],
    "failed_assembly_reason": "slurm_comma_delimited_export_truncated_audited_arms_to_p0",
    "repair_scope": "run_root_reuse_and_slurm_list_serialization_only",
    "list_transport": "colon_delimited_in_slurm_environment_decoded_before_stage",
    "launchers": launchers,
    "raw64_sha256": raw64,
    "training_terminal_sha256": training_terminals,
    "raw64_bytes_changed": False,
    "training_or_checkpoint_changed": False,
    "model_data_prompt_seed_optimizer_scheduler_ledger_evaluator_gate_changed": False,
    "smact4_executed_on_a800": False,
    "broad_tests_repeated": False,
}
(root / "V12_REPAIR_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

(
  cd "${RUN_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum \
    prepare_v12_slurm_list_serialization_repair_on_a800.sh \
    submission_record.json \
    V12_REPAIR_RECORD.json
) > "${RUN_ROOT}/V12_SHA256.txt"
(cd "${RUN_ROOT}" && sha256sum -c V12_SHA256.txt)
touch "${RUN_ROOT}/status/preparation_SUCCESS"
chmod 400 \
  "${RUN_ROOT}"/launchers/* \
  "${RUN_ROOT}/prepare_v12_slurm_list_serialization_repair_on_a800.sh" \
  "${RUN_ROOT}/submission_record.json" \
  "${RUN_ROOT}/submission_record.sha256" \
  "${RUN_ROOT}/V12_REPAIR_RECORD.json" \
  "${RUN_ROOT}/V12_SHA256.txt"
chmod 500 "${RUN_ROOT}/launchers"
cat "${RUN_ROOT}/V12_REPAIR_RECORD.json"

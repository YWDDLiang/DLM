#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V15_SOURCE="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_v15_raw256_assembly_c4c969b.sh"
V16_SOURCE="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_v16_sacct_array_task_repair_9f171f2.sh"
V15_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_path_repair_v15"
V16_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_array_task_repair_v16"
V17_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_array_task_run_root_literal_repair_v17"
GENERATION_PARENT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_submission_cwd_repair_v14"
AUDIT_STAGING="${PROJECT_ROOT}/runs/transfer_raw256_smact4_audit_ed17201d"
AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
EXPECTED_SELF_SHA256="${1:?expected V17 generator SHA256}"
EXPECTED_V15_SOURCE_SHA256=274321f81900022135cc1d510f7a8825522b36d441516b70499392fdcaf36000
EXPECTED_V16_SOURCE_SHA256=779b9b58f8103a85de7b9531f6b23ec3fada79163120785e366e91ac37fc2988
EXPECTED_V15_SACCT_SHA256=f7c8e3ca3d96c71b13328e237cf743f4a63a36f997570b306b8e472b21df0d74
SELF="$(realpath "${BASH_SOURCE[0]}")"
V16_STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v16_sacct_array_task_repair_${EXPECTED_V16_SOURCE_SHA256}"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v17_sacct_array_task_run_root_literal_repair_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_v17_raw256_assembly_on_a800.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V15_SOURCE}"
test "$(sha256sum "${V15_SOURCE}" | cut -d' ' -f1)" = "${EXPECTED_V15_SOURCE_SHA256}"
test -f "${V16_SOURCE}"
test "$(sha256sum "${V16_SOURCE}" | cut -d' ' -f1)" = "${EXPECTED_V16_SOURCE_SHA256}"
test ! -e "${V15_RUN_ROOT}"
test ! -e "${V16_RUN_ROOT}"
test ! -e "${V17_RUN_ROOT}"
test -d "${V16_STAGE_ROOT}"
test -f "${V16_STAGE_ROOT}/$(basename "${V16_SOURCE}")"
test "$(sha256sum "${V16_STAGE_ROOT}/$(basename "${V16_SOURCE}")" | cut -d' ' -f1)" = \
  "${EXPECTED_V16_SOURCE_SHA256}"
test ! -e "${V16_STAGE_ROOT}/prepare_and_submit_v16_raw256_assembly_on_a800.sh"
test ! -e "${V16_STAGE_ROOT}/V16_GENERATION_RECORD.json"
test ! -e "${V16_STAGE_ROOT}/V16_STAGE_SHA256.txt"
test ! -e "${STAGE_ROOT}"
test -d "${AUDIT_STAGING}"
test ! -e "${AUDIT_ROOT}"
test -f "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt"
test "$(sha256sum "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V15_SACCT_SHA256}"
test "$(grep -Ec '^31236_[01][|]' "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt")" -eq 0
test "$(grep -Ec '^31236[|]COMPLETED[|]0:0$' "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt")" -eq 1

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"

export V15_SOURCE GENERATED V17_RUN_ROOT
export EXPECTED_V15_SACCT_SHA256 EXPECTED_V16_SOURCE_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V15_SOURCE"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen V15 source")

old_run = "20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_path_repair_v15"
new_run = Path(os.environ["V17_RUN_ROOT"]).name
outer_before = 'sacct -n -X -j "${EXPECTED_JOB_ID}"'
outer_after = 'sacct -n -j "${EXPECTED_JOB_ID}"'
if text.count(old_run) != 2:
    raise SystemExit("V15 run-name identity mismatch")
if text.count(outer_before) != 1:
    raise SystemExit("V15 outer sacct query identity mismatch")
if text.count("v15") != 18 or text.count("V15") != 6:
    raise SystemExit("V15 version-marker census mismatch")

text = text.replace(old_run, new_run)
text = text.replace(outer_before, outer_after, 1)
text = text.replace("v15", "v17").replace("V15", "V17")
if old_run in text or outer_before in text or "v15" in text or "V15" in text:
    raise SystemExit("stale V15 marker remains before V17 evidence insertion")

old_scope = (
    '    "repair_scope": '
    '"assembly_run_root_and_v14_submission_schema_adapter_only",\n'
)
new_scope = (
    '    "failed_v15_sacct_sha256": '
    f'"{os.environ["EXPECTED_V15_SACCT_SHA256"]}",\n'
    '    "failed_v15_reason": "sacct_-X_omitted_array_task_rows",\n'
    '    "failed_v16_wrapper_sha256": '
    f'"{os.environ["EXPECTED_V16_SOURCE_SHA256"]}",\n'
    '    "failed_v16_reason": "generator_run_root_literal_assertion_mismatch",\n'
    '    "repair_scope": '
    '"assembly_path_schema_adapter_plus_outer_inner_sacct_array_task_visibility_and_run_root_literal_assertion_only",\n'
)
if text.count(old_scope) != 1:
    raise SystemExit("V17 repair-record insertion identity mismatch")
text = text.replace(old_scope, new_scope, 1)

inner_anchor = (
    '    if source_name == "submit_assemble256_once.sh":\n'
    '        execution_line = (\n'
)
inner_replacement = (
    '    if source_name == "submit_assemble256_once.sh":\n'
    '        inner_sacct_before = \'sacct -n -X -j "${PLANNER_JOB_ID}"\'\n'
    '        inner_sacct_after = \'sacct -n -j "${PLANNER_JOB_ID}"\'\n'
    '        if text.count(inner_sacct_before) != 1:\n'
    '            raise SystemExit("inner submit sacct identity mismatch")\n'
    '        text = text.replace(inner_sacct_before, inner_sacct_after, 1)\n'
    '        if text.count(inner_sacct_before) != 0 or text.count(inner_sacct_after) != 1:\n'
    '            raise SystemExit("inner submit sacct replacement mismatch")\n'
    '        execution_line = (\n'
)
if text.count(inner_anchor) != 1:
    raise SystemExit("V17 inner submit insertion identity mismatch")
text = text.replace(inner_anchor, inner_replacement, 1)

launcher_check_marker = '\nexport GENERATION_SACCT_SHA\n'
launcher_checks = (
    '\ntest "$(grep -Fc \'sacct -n -j "${PLANNER_JOB_ID}"\' '
    '"${RUN_ROOT}/launchers/submit_assemble256_v17_once.sh")" -eq 1\n'
    'test "$(grep -Fc \'sacct -n -X -j "${PLANNER_JOB_ID}"\' '
    '"${RUN_ROOT}/launchers/submit_assemble256_v17_once.sh")" -eq 0\n'
)
if text.count(launcher_check_marker) != 1:
    raise SystemExit("V17 launcher-check insertion identity mismatch")
text = text.replace(launcher_check_marker, launcher_checks + launcher_check_marker, 1)

expected_root = f'RUN_ROOT="${{PROJECT_ROOT}}/runs/{new_run}"'
if text.count(expected_root) != 1:
    raise SystemExit("V17 run-root literal identity mismatch")
if text.count(outer_after) != 1 or text.count(outer_before) != 0:
    raise SystemExit("V17 outer sacct replacement mismatch")
if text.count('inner_sacct_before = \'sacct -n -X -j "${PLANNER_JOB_ID}"\'') != 1:
    raise SystemExit("V17 inner old-query guard mismatch")
if text.count('inner_sacct_after = \'sacct -n -j "${PLANNER_JOB_ID}"\'') != 1:
    raise SystemExit("V17 inner new-query guard mismatch")
if text.count("submit_assemble256_v17_once.sh") < 2:
    raise SystemExit("V17 generated submit identity mismatch")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'sacct -n -j "${EXPECTED_JOB_ID}"' "${GENERATED}"
if grep -Fq 'sacct -n -X -j "${EXPECTED_JOB_ID}"' "${GENERATED}"; then
  echo 'stale outer allocation-only sacct query remains' >&2
  exit 3
fi
grep -Fq 'inner_sacct_before' "${GENERATED}"
grep -Fq 'inner_sacct_after' "${GENERATED}"
grep -Fq 'submit_assemble256_v17_once.sh' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/V17_GENERATION_RECORD.json" <<EOF
{
  "schema": "h1_chemistry_first_v17_outer_inner_sacct_and_run_root_literal_repair_generator",
  "status": "pass",
  "v15_source_sha256": "${EXPECTED_V15_SOURCE_SHA256}",
  "failed_v15_sacct_sha256": "${EXPECTED_V15_SACCT_SHA256}",
  "failed_v15_reason": "sacct_-X_omitted_array_task_rows",
  "failed_v16_wrapper_sha256": "${EXPECTED_V16_SOURCE_SHA256}",
  "failed_v16_reason": "generator_run_root_literal_assertion_mismatch",
  "generated_v17_sha256": "${GENERATED_SHA256}",
  "repair_scope": "outer_and_inner_sacct_array_task_visibility_plus_run_root_literal_assertion_only",
  "science_contract_changed": false,
  "smact4_executed_on_a800": false,
  "broad_tests_repeated": false
}
EOF
(
  cd "${STAGE_ROOT}"
  sha256sum \
    "$(basename "${SELF}")" \
    "$(basename "${GENERATED}")" \
    V17_GENERATION_RECORD.json
) > "${STAGE_ROOT}/V17_STAGE_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c V17_STAGE_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

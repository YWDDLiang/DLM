#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V17_STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v17_sacct_array_task_run_root_literal_repair_d51e924d257407075573f1c2c4e0f049b1039b3fe6490c6495e2e76e18068eb8"
V17_SOURCE="${V17_STAGE_ROOT}/prepare_and_submit_v17_raw256_assembly_on_a800.sh"
V17_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_array_task_run_root_literal_repair_v17"
V18_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_jobid_field_repair_v18"
GENERATION_PARENT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_submission_cwd_repair_v14"
AUDIT_STAGING="${PROJECT_ROOT}/runs/transfer_raw256_smact4_audit_ed17201d"
AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
EXPECTED_SELF_SHA256="${1:?expected V18 generator SHA256}"
EXPECTED_V17_SOURCE_SHA256=070af23a6c022664cc5e0ec62f5edb60f5ffa34167b6491794fe66d7a574b820
EXPECTED_V17_STAGE_SHA256=e855c94a0b622d8eec2072657ad758e50d77391f73e91f779ee5234b4568ed42
EXPECTED_V17_FAILURE_SHA256=cdf80bb5659e2331bbde0270fddd7e41f8e4b73bb6fa5d1fe5999190b61822e8
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v18_sacct_jobid_field_repair_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_v18_raw256_assembly_on_a800.sh"
JOBID_PROBE="${STAGE_ROOT}/sacct_jobid_probe.txt"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -d "${V17_STAGE_ROOT}"
test -f "${V17_SOURCE}"
test "$(sha256sum "${V17_SOURCE}" | cut -d' ' -f1)" = "${EXPECTED_V17_SOURCE_SHA256}"
test "$(sha256sum "${V17_STAGE_ROOT}/V17_STAGE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V17_STAGE_SHA256}"
(cd "${V17_STAGE_ROOT}" && sha256sum -c V17_STAGE_SHA256.txt)
test ! -e "${V17_RUN_ROOT}"
test ! -e "${V18_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"
test -d "${AUDIT_STAGING}"
test ! -e "${AUDIT_ROOT}"
test -f "${GENERATION_PARENT}/status/sacct_planner256_before_v17.txt"
test "$(sha256sum "${GENERATION_PARENT}/status/sacct_planner256_before_v17.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V17_FAILURE_SHA256}"
test "$(grep -Ec '^31236_[01][|]' "${GENERATION_PARENT}/status/sacct_planner256_before_v17.txt")" -eq 0
test "$(grep -Ec '^3123[67][|]COMPLETED[|]0:0$' "${GENERATION_PARENT}/status/sacct_planner256_before_v17.txt")" -eq 2

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
sacct -n -j 31236 --format=JobID,JobIDRaw,State,ExitCode -P > "${JOBID_PROBE}"
for task in 0 1; do
  row="$(awk -F'|' -v wanted="31236_${task}" '$1 == wanted {print $3 "|" $4}' "${JOBID_PROBE}")"
  test "${row}" = 'COMPLETED|0:0'
done
test "$(grep -Ec '^31236_[01][|][0-9]+[|]COMPLETED[|]0:0$' "${JOBID_PROBE}")" -eq 2
JOBID_PROBE_SHA256="$(sha256sum "${JOBID_PROBE}" | cut -d' ' -f1)"

export V17_SOURCE GENERATED V18_RUN_ROOT
export EXPECTED_V17_FAILURE_SHA256 EXPECTED_V17_SOURCE_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V17_SOURCE"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen V17 source")

old_run = "20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_array_task_run_root_literal_repair_v17"
new_run = Path(os.environ["V18_RUN_ROOT"]).name
old_field = "--format=JobIDRaw,State,ExitCode -P"
new_field = "--format=JobID,State,ExitCode -P"
if text.count(old_run) != 2:
    raise SystemExit("V17 run-name identity mismatch")
if text.count(old_field) != 1:
    raise SystemExit("V17 outer sacct field identity mismatch")
if text.count("v17") < 8 or text.count("V17") < 5:
    raise SystemExit("V17 version-marker census mismatch")

text = text.replace(old_run, new_run)
text = text.replace(old_field, new_field, 1)
text = text.replace("v17", "v18").replace("V17", "V18")
if old_run in text or old_field in text or "v17" in text or "V17" in text:
    raise SystemExit("stale V17 marker remains before V18 evidence insertion")

old_scope = (
    '    "repair_scope": '
    '"assembly_path_schema_adapter_plus_outer_inner_sacct_array_task_visibility_and_run_root_literal_assertion_only",\n'
)
new_scope = (
    '    "failed_v17_sacct_sha256": '
    f'"{os.environ["EXPECTED_V17_FAILURE_SHA256"]}",\n'
    '    "failed_v17_source_sha256": '
    f'"{os.environ["EXPECTED_V17_SOURCE_SHA256"]}",\n'
    '    "failed_v17_reason": "JobIDRaw_exposed_internal_allocation_ids_not_logical_array_task_ids",\n'
    '    "repair_scope": '
    '"assembly_path_schema_adapter_plus_outer_inner_sacct_visibility_run_root_literal_and_outer_JobID_field_only",\n'
)
if text.count(old_scope) != 1:
    raise SystemExit("V18 repair-record insertion identity mismatch")
text = text.replace(old_scope, new_scope, 1)

expected_root = f'RUN_ROOT="${{PROJECT_ROOT}}/runs/{new_run}"'
if text.count(expected_root) != 1:
    raise SystemExit("V18 run-root literal identity mismatch")
if text.count(new_field) != 1 or text.count(old_field) != 0:
    raise SystemExit("V18 outer sacct field replacement mismatch")
if text.count('inner_sacct_before = \'sacct -n -X -j "${PLANNER_JOB_ID}"\'') != 1:
    raise SystemExit("V18 inner old-query guard mismatch")
if text.count('inner_sacct_after = \'sacct -n -j "${PLANNER_JOB_ID}"\'') != 1:
    raise SystemExit("V18 inner new-query guard mismatch")
if text.count("submit_assemble256_v18_once.sh") < 2:
    raise SystemExit("V18 generated submit identity mismatch")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'sacct -n -j "${EXPECTED_JOB_ID}" --format=JobID,State,ExitCode -P' "${GENERATED}"
if grep -Fq -- '--format=JobIDRaw,State,ExitCode -P' "${GENERATED}"; then
  echo 'stale outer JobIDRaw field remains' >&2
  exit 3
fi
grep -Fq 'inner_sacct_before' "${GENERATED}"
grep -Fq 'inner_sacct_after' "${GENERATED}"
grep -Fq 'submit_assemble256_v18_once.sh' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/V18_GENERATION_RECORD.json" <<EOF
{
  "schema": "h1_chemistry_first_v18_sacct_jobid_field_repair_generator",
  "status": "pass",
  "v17_source_sha256": "${EXPECTED_V17_SOURCE_SHA256}",
  "failed_v17_sacct_sha256": "${EXPECTED_V17_FAILURE_SHA256}",
  "failed_v17_reason": "JobIDRaw_exposed_internal_allocation_ids_not_logical_array_task_ids",
  "jobid_probe_sha256": "${JOBID_PROBE_SHA256}",
  "generated_v18_sha256": "${GENERATED_SHA256}",
  "repair_scope": "outer_sacct_JobIDRaw_to_JobID_only",
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
    "$(basename "${JOBID_PROBE}")" \
    V18_GENERATION_RECORD.json
) > "${STAGE_ROOT}/V18_STAGE_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c V18_STAGE_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V21_SOURCE="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_v21_raw_source_62ebd2f.sh"
V21_PARTIAL_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v20_raw256_assembly_raw_source_identity_repair_v21"
V22_SOURCE="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_v22_guard_anchor_d84d0b4.sh"
V22_STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v22_guard_anchor_generator_3fc35c7facd4d23b66152f1c57cdb3548461a7c1567a8324a5ed6a80909e2434"
V22_FAILED_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v21_raw256_assembly_guard_anchor_repair_v22"
V23_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v20_raw256_assembly_guard_anchor_repair_v23"
EXPECTED_SELF_SHA256="${1:?expected V23 generator SHA256}"
EXPECTED_V21_SOURCE_SHA256=fc9424fd3fad24eae87889fb021a00339c063addf9e6234430e1ec5d3cd9811c
EXPECTED_V22_SOURCE_SHA256=3fc35c7facd4d23b66152f1c57cdb3548461a7c1567a8324a5ed6a80909e2434
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v23_root_substitution_generator_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_v23_root_substitution_repair_generated.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V21_SOURCE}"
test "$(sha256sum "${V21_SOURCE}" | cut -d' ' -f1)" = \
  "${EXPECTED_V21_SOURCE_SHA256}"
test -d "${V21_PARTIAL_ROOT}"
test -d "${V21_PARTIAL_ROOT}/launchers"
test -z "$(find "${V21_PARTIAL_ROOT}/launchers" -mindepth 1 -maxdepth 1 -print -quit)"
test ! -e "${V21_PARTIAL_ROOT}/status/preparation_SUCCESS"
test ! -e "${V21_PARTIAL_ROOT}/status/submitted_assemble256_job_id.txt"
test -f "${V22_SOURCE}"
test "$(sha256sum "${V22_SOURCE}" | cut -d' ' -f1)" = \
  "${EXPECTED_V22_SOURCE_SHA256}"
test -d "${V22_STAGE_ROOT}"
test "$(find "${V22_STAGE_ROOT}" -mindepth 1 -maxdepth 1 -type f -printf '%f\n')" = \
  "$(basename "${V22_SOURCE}")"
test ! -e "${V22_FAILED_RUN_ROOT}"
test ! -e "${V23_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
export V21_SOURCE GENERATED V23_RUN_ROOT
export EXPECTED_V21_SOURCE_SHA256 EXPECTED_V22_SOURCE_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V21_SOURCE"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen V21 source")

old_root = (
    "20260809_h1_chemistry_first_sft_v2_v20_raw256_assembly_"
    "raw_source_identity_repair_v21"
)
placeholder = "__V23_FINAL_RUN_ROOT_PLACEHOLDER__"
new_root = Path(os.environ["V23_RUN_ROOT"]).name
old_guard = (
    "        audit_guard = "
    "'test ! -e \"${REPAIR_ROOT}/assembly256_submission_record.json\"\\n'\n"
)
new_guard = (
    "        audit_guard = "
    "'test ! -e \"${RUN_ROOT}/assembly256_submission_record.json\"\\n'\n"
)
old_scope = (
    '    "repair_scope": '
    '"separate_raw_generation_and_assembly_source_identities",\n'
)
new_scope = (
    '    "failed_v21_source_sha256": '
    f'"{os.environ["EXPECTED_V21_SOURCE_SHA256"]}",\n'
    '    "failed_v22_source_sha256": '
    f'"{os.environ["EXPECTED_V22_SOURCE_SHA256"]}",\n'
    '    "failed_v21_reason": '
    '"generator_guard_expected_REPAIR_ROOT_but_launcher_uses_RUN_ROOT",\n'
    '    "failed_v22_reason": '
    '"new_root_parent_version_marker_was_rewritten_a_second_time",\n'
    '    "repair_scope": '
    '"separate_raw_and_assembly_source_identities_with_exact_guard_and_root_substitution",\n'
)
if text.count(old_root) != 1:
    raise SystemExit("V21 run-root identity mismatch")
if text.count(old_guard) != 1:
    raise SystemExit("V21 guard-source identity mismatch")
if text.count(old_scope) != 1:
    raise SystemExit("V21 repair-scope identity mismatch")

text = text.replace(old_root, placeholder, 1)
text = text.replace(old_guard, new_guard, 1)
text = text.replace("V21", "V23").replace("v21", "v23")
if old_root in text or old_guard in text or "V21" in text or "v21" in text:
    raise SystemExit("stale V21 implementation marker remains")
if text.count(placeholder) != 1:
    raise SystemExit("V23 root placeholder census mismatch")
text = text.replace(placeholder, new_root, 1)
text = text.replace(old_scope, new_scope, 1)

expected_root = f'V23_ROOT="${{PROJECT_ROOT}}/runs/{new_root}"'
if text.count(expected_root) != 1:
    raise SystemExit("V23 run-root literal mismatch")
if text.count(
    'audit_guard = \'test ! -e "${RUN_ROOT}/assembly256_submission_record.json"\\n\''
) != 1:
    raise SystemExit("V23 guard anchor mismatch")
if text.count('"failed_v21_source_sha256"') != 1:
    raise SystemExit("V23 failed-V21 evidence missing")
if text.count('"failed_v22_source_sha256"') != 1:
    raise SystemExit("V23 failed-V22 evidence missing")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'submit_assemble256_v23_once.sh' "${GENERATED}"
grep -Fq '#SBATCH --partition=normal' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/V23_GENERATION_RECORD.json" <<EOF
{
  "schema": "h1_chemistry_first_v23_root_substitution_generator",
  "status": "pass",
  "failed_v21_source_sha256": "${EXPECTED_V21_SOURCE_SHA256}",
  "failed_v22_source_sha256": "${EXPECTED_V22_SOURCE_SHA256}",
  "failed_v22_reason": "new_root_parent_version_marker_was_rewritten_a_second_time",
  "generated_v23_sha256": "${GENERATED_SHA256}",
  "repair_scope": "placeholder_protected_final_run_root_substitution_only",
  "v21_or_v22_reexecuted": false,
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
    V23_GENERATION_RECORD.json
) > "${STAGE_ROOT}/V23_GENERATOR_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c V23_GENERATOR_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

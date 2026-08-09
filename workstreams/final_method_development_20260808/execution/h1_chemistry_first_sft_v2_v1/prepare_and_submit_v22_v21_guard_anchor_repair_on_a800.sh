#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V21_SOURCE="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_v21_raw_source_62ebd2f.sh"
V21_PARTIAL_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v20_raw256_assembly_raw_source_identity_repair_v21"
V22_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v21_raw256_assembly_guard_anchor_repair_v22"
EXPECTED_SELF_SHA256="${1:?expected V22 generator SHA256}"
EXPECTED_V21_SOURCE_SHA256=fc9424fd3fad24eae87889fb021a00339c063addf9e6234430e1ec5d3cd9811c
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v22_guard_anchor_generator_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_v22_guard_anchor_repair_generated.sh"

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
test ! -e "${V21_PARTIAL_ROOT}/assembly256_submission_record.json"
test ! -e "${V21_PARTIAL_ROOT}/V21_REPAIR_RECORD.json"
test ! -e "${V22_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
export V21_SOURCE GENERATED V22_RUN_ROOT EXPECTED_V21_SOURCE_SHA256
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
new_root = Path(os.environ["V22_RUN_ROOT"]).name
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
    '    "failed_v21_reason": '
    '"generator_guard_expected_REPAIR_ROOT_but_launcher_uses_RUN_ROOT",\n'
    '    "repair_scope": '
    '"separate_raw_generation_and_assembly_source_identities_plus_exact_guard_anchor",\n'
)
if text.count(old_root) != 1:
    raise SystemExit("V21 run-root identity mismatch")
if text.count(old_guard) != 1:
    raise SystemExit("V21 guard-source identity mismatch")
if text.count(old_scope) != 1:
    raise SystemExit("V21 repair-scope identity mismatch")

text = text.replace(old_root, new_root, 1)
text = text.replace(old_guard, new_guard, 1)
text = text.replace("V21", "V22").replace("v21", "v22")
if old_root in text or old_guard in text or "V21" in text or "v21" in text:
    raise SystemExit("stale V21 marker remains")
text = text.replace(old_scope, new_scope, 1)
expected_root = f'V22_ROOT="${{PROJECT_ROOT}}/runs/{new_root}"'
if text.count(expected_root) != 1:
    raise SystemExit("V22 run-root literal mismatch")
if text.count(
    'audit_guard = \'test ! -e "${RUN_ROOT}/assembly256_submission_record.json"\\n\''
) != 1:
    raise SystemExit("V22 guard anchor mismatch")
if text.count('"failed_v21_source_sha256"') != 1:
    raise SystemExit("V22 failed-V21 evidence missing")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'submit_assemble256_v22_once.sh' "${GENERATED}"
grep -Fq 'audit_guard = '\''test ! -e "${RUN_ROOT}/assembly256_submission_record.json"\n'\''' \
  "${GENERATED}"
grep -Fq '#SBATCH --partition=normal' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/V22_GENERATION_RECORD.json" <<EOF
{
  "schema": "h1_chemistry_first_v22_guard_anchor_generator",
  "status": "pass",
  "failed_v21_source_sha256": "${EXPECTED_V21_SOURCE_SHA256}",
  "failed_v21_reason": "generator_guard_expected_REPAIR_ROOT_but_launcher_uses_RUN_ROOT",
  "generated_v22_sha256": "${GENERATED_SHA256}",
  "repair_scope": "REPAIR_ROOT_to_RUN_ROOT_generator_anchor_only",
  "v21_reexecuted": false,
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
    V22_GENERATION_RECORD.json
) > "${STAGE_ROOT}/V22_GENERATOR_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c V22_GENERATOR_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

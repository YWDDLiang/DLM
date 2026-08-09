#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V2_WRAPPER="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_b0_v2_test_package_repair_eec7f4d.sh"
V2_FAILED_STAGE="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_b0_v2_test_package_repair_380243847ba4011c6e3bdd9734e054c295f438ab621b94d22086d1eb8963a176"
V2_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v2"
V3_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v3"
EXPECTED_SELF_SHA256="${1:?expected B0-v3 generator SHA256}"
EXPECTED_V2_WRAPPER_SHA256=380243847ba4011c6e3bdd9734e054c295f438ab621b94d22086d1eb8963a176
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_b0_v3_tar126_generator_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_b0_v3_tar126_repair_generated.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V2_WRAPPER}"
test "$(sha256sum "${V2_WRAPPER}" | cut -d' ' -f1)" = \
  "${EXPECTED_V2_WRAPPER_SHA256}"
test -d "${V2_FAILED_STAGE}/source"
test -f "${V2_FAILED_STAGE}/$(basename "${V2_WRAPPER}")"
test "$(sha256sum "${V2_FAILED_STAGE}/$(basename "${V2_WRAPPER}")" | cut -d' ' -f1)" = \
  "${EXPECTED_V2_WRAPPER_SHA256}"
test ! -e "${V2_FAILED_STAGE}/dlm_state_panels_b0_v2_source.tar.gz"
test ! -e "${V2_FAILED_STAGE}/B0_V2_REPAIR_RECORD.json"
test ! -e "${V2_RUN_ROOT}"
test ! -e "${V3_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
export V2_WRAPPER GENERATED EXPECTED_V2_WRAPPER_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V2_WRAPPER"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen B0-v2 wrapper")

old_tar = """tar --sort=name --mtime='UTC 2026-08-09' --owner=0 --group=0 --numeric-owner \\
  -czf "${V3_ARCHIVE}" -C "${SOURCE_ROOT}" ."""
new_tar = 'tar -czf "${V3_ARCHIVE}" -C "${SOURCE_ROOT}" .'
if text.count("V2") != 17 or text.count("v2") != 10:
    raise SystemExit("B0-v2 version-marker census mismatch")

text = text.replace("V2", "V3").replace("v2", "v3")
text = text.replace(
    "transfer_dlm_state_panels_b0_v3_test_package_repair_",
    "transfer_dlm_state_panels_b0_v3_tar126_repair_",
    1,
)
text = text.replace(
    "evidence_first_dlm_state_panels_b0_v3_test_package_repair",
    "evidence_first_dlm_state_panels_b0_v3_tar126_repair",
    1,
)
if text.count(old_tar) != 1:
    raise SystemExit("B0-v2 GNU tar command identity mismatch")
text = text.replace(old_tar, new_tar, 1)
if "V2" in text or "v2" in text:
    raise SystemExit("stale B0-v2 marker remains before failure-evidence insertion")

old_record = (
    '  "failed_v1_reason": "tests_package_marker_omitted_from_frozen_archive",\n'
)
new_record = (
    old_record
    + '  "failed_v2_reason": "gnu_tar_1_26_missing_sort_option",\n'
    + '  "failed_v2_wrapper_sha256": "'
    + os.environ["EXPECTED_V2_WRAPPER_SHA256"]
    + '",\n'
    + '  "archive_reproducibility_flags_changed": true,\n'
)
if text.count(old_record) != 1:
    raise SystemExit("B0-v3 repair-record insertion identity mismatch")
text = text.replace(old_record, new_record, 1)

if text.count('"schema": "evidence_first_dlm_state_panels_v1"') != 1:
    raise SystemExit("B0 CONFIG schema changed")
if text.count('"schema": "evidence_first_dlm_state_panel_submission_v1"') != 1:
    raise SystemExit("B0 submission schema changed")
if text.count(new_tar) != 1 or "--sort=name" in text:
    raise SystemExit("B0-v3 tar compatibility repair mismatch")
if text.count("20260809_h1_dlm_state_panels_b0_v3") != 2:
    raise SystemExit("B0-v3 run-root identity mismatch")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'tar -czf "${V3_ARCHIVE}" -C "${SOURCE_ROOT}" .' "${GENERATED}"
if grep -Fq -- '--sort=name' "${GENERATED}"; then
  echo 'stale unsupported tar flag remains in B0-v3' >&2
  exit 3
fi
grep -Fq '"schema": "evidence_first_dlm_state_panels_v1"' "${GENERATED}"
grep -Fq '"schema": "evidence_first_dlm_state_panel_submission_v1"' "${GENERATED}"
grep -Fq '#SBATCH --partition=gpu' "${GENERATED}"
if grep -Fq 'gpu_long' "${GENERATED}"; then
  echo 'forbidden gpu_long partition in B0-v3' >&2
  exit 3
fi

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/B0_V3_GENERATION_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_state_panels_b0_v3_tar126_generator",
  "status": "pass",
  "failed_v2_wrapper_sha256": "${EXPECTED_V2_WRAPPER_SHA256}",
  "failed_v2_reason": "gnu_tar_1_26_missing_sort_option",
  "generated_v3_sha256": "${GENERATED_SHA256}",
  "repair_scope": "archive_command_tar126_compatibility_only",
  "science_contract_changed": false,
  "schema_changed": false,
  "training": false,
  "sun": false,
  "automatic_b3_submission": false,
  "broad_tests_repeated": false
}
EOF
(
  cd "${STAGE_ROOT}"
  sha256sum \
    "$(basename "${SELF}")" \
    "$(basename "${GENERATED}")" \
    B0_V3_GENERATION_RECORD.json
) > "${STAGE_ROOT}/B0_V3_GENERATOR_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c B0_V3_GENERATOR_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

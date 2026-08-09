#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V3_GENERATOR_STAGE="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_b0_v3_tar126_generator_a337e40116803ea06b61dac592b08eb8fd51eadd170933bd9bfbd0fc39d7136a"
V3_SOURCE="${V3_GENERATOR_STAGE}/prepare_and_submit_b0_v3_tar126_repair_generated.sh"
V3_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v3"
V4_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v4"
EXPECTED_SELF_SHA256="${1:?expected B0-v4 generator SHA256}"
EXPECTED_V3_SOURCE_SHA256=cf8f554d4b1c8bc274cd1ba217186c0e36ebb8bd52ee85cc44eafef6cfeb98a8
EXPECTED_V3_GENERATOR_SHA256=a337e40116803ea06b61dac592b08eb8fd51eadd170933bd9bfbd0fc39d7136a
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_b0_v4_outer_guard_generator_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_b0_v4_outer_guard_repair_generated.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V3_SOURCE}"
test "$(sha256sum "${V3_SOURCE}" | cut -d' ' -f1)" = \
  "${EXPECTED_V3_SOURCE_SHA256}"
test -f "${V3_GENERATOR_STAGE}/transfer_prepare_and_submit_b0_v3_tar126_repair_d9f8aa1.sh"
test "$(sha256sum "${V3_GENERATOR_STAGE}/transfer_prepare_and_submit_b0_v3_tar126_repair_d9f8aa1.sh" | cut -d' ' -f1)" = \
  "${EXPECTED_V3_GENERATOR_SHA256}"
test ! -e "${V3_GENERATOR_STAGE}/B0_V3_GENERATION_RECORD.json"
test ! -e "${V3_RUN_ROOT}"
test ! -e "${V4_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
export V3_SOURCE GENERATED EXPECTED_V3_SOURCE_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V3_SOURCE"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen B0-v3 generated wrapper")
if text.count("V3") != 17 or text.count("v3") != 10:
    raise SystemExit("B0-v3 version-marker census mismatch")

text = text.replace("V3", "V4").replace("v3", "v4")
if "V3" in text or "v3" in text:
    raise SystemExit("stale B0-v3 marker remains before failure-evidence insertion")

old_record = '  "archive_reproducibility_flags_changed": true,\n'
new_record = (
    old_record
    + '  "failed_v3_reason": "outer_gpu_long_guard_matched_its_own_guard_text",\n'
    + '  "failed_v3_generated_sha256": "'
    + os.environ["EXPECTED_V3_SOURCE_SHA256"]
    + '",\n'
    + '  "outer_guard_only_changed": true,\n'
)
if text.count(old_record) != 1:
    raise SystemExit("B0-v4 repair-record insertion identity mismatch")
text = text.replace(old_record, new_record, 1)

if text.count('"schema": "evidence_first_dlm_state_panels_v1"') != 1:
    raise SystemExit("B0 CONFIG schema changed")
if text.count('"schema": "evidence_first_dlm_state_panel_submission_v1"') != 1:
    raise SystemExit("B0 submission schema changed")
if text.count('tar -czf "${V4_ARCHIVE}" -C "${SOURCE_ROOT}" .') != 1:
    raise SystemExit("B0-v4 tar compatibility command mismatch")
if "--sort=name" in text:
    raise SystemExit("unsupported tar flag returned in B0-v4")
if text.count("20260809_h1_dlm_state_panels_b0_v4") != 2:
    raise SystemExit("B0-v4 run-root identity mismatch")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'tar -czf "${V4_ARCHIVE}" -C "${SOURCE_ROOT}" .' "${GENERATED}"
grep -Fq '"schema": "evidence_first_dlm_state_panels_v1"' "${GENERATED}"
grep -Fq '"schema": "evidence_first_dlm_state_panel_submission_v1"' "${GENERATED}"
grep -Fq "if grep -Fq 'gpu_long'" "${GENERATED}"
grep -Fq '"${EXECUTION_DIR}/state_panels.sbatch"' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/B0_V4_GENERATION_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_state_panels_b0_v4_outer_guard_generator",
  "status": "pass",
  "failed_v3_generated_sha256": "${EXPECTED_V3_SOURCE_SHA256}",
  "failed_v3_reason": "outer_gpu_long_guard_matched_its_own_guard_text",
  "generated_v4_sha256": "${GENERATED_SHA256}",
  "repair_scope": "remove_outer_self_matching_gpu_long_guard_only",
  "inner_actual_sbatch_gpu_long_guard_preserved": true,
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
    B0_V4_GENERATION_RECORD.json
) > "${STAGE_ROOT}/B0_V4_GENERATOR_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c B0_V4_GENERATOR_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SOURCE_ARCHIVE="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_v1_192d6c2.tar.gz"
V1_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v1"
V2_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v2"
EXPECTED_SELF_SHA256="${1:?expected B0-v2 repair SHA256}"
EXPECTED_SOURCE_ARCHIVE_SHA256=d3a78c15022fefec6a62cedeb8bf3e18e28743e2495872936d6d491389ce80ba
EXPECTED_V1_PREPARE_FAILED_SHA256=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865
EXPECTED_V1_FOCUSED_TEST_SHA256=d02205fd5d206c793e5e41f502a2233d6e021d607464976c599c1ef21014d763
EXPECTED_TESTS_INIT_SHA256=8dc626f6f2daece500b7977a074ae24f093ebe46bfef81d9839bcca2f2d6e752
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_state_panels_b0_v2_test_package_repair_${EXPECTED_SELF_SHA256}"
SOURCE_ROOT="${STAGE_ROOT}/source"
EXECUTION_REL=workstreams/final_method_development_20260808/execution/dlm_state_panels_v1
EXECUTION_DIR="${SOURCE_ROOT}/${EXECUTION_REL}"
V2_ARCHIVE="${STAGE_ROOT}/dlm_state_panels_b0_v2_source.tar.gz"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${SOURCE_ARCHIVE}"
test "$(sha256sum "${SOURCE_ARCHIVE}" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_ARCHIVE_SHA256}"
test -d "${V1_RUN_ROOT}"
test "$(sha256sum "${V1_RUN_ROOT}/status/PREPARE_FAILED" | cut -d' ' -f1)" = \
  "${EXPECTED_V1_PREPARE_FAILED_SHA256}"
test "$(sha256sum "${V1_RUN_ROOT}/status/focused_test.log" | cut -d' ' -f1)" = \
  "${EXPECTED_V1_FOCUSED_TEST_SHA256}"
grep -Fq "No module named 'tests.test_planned_corruption'" \
  "${V1_RUN_ROOT}/status/focused_test.log"
test ! -e "${V1_RUN_ROOT}/status/submission.lock"
test ! -e "${V1_RUN_ROOT}/status/submission_record.json"
test ! -e "${V2_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir -p "${SOURCE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
tar -xf "${SOURCE_ARCHIVE}" -C "${SOURCE_ROOT}"
test -f "${EXECUTION_DIR}/SOURCE_FILES.txt"
test ! -e "${SOURCE_ROOT}/tests/__init__.py"
test "$(grep -Fxc 'tests/test_planned_corruption.py' "${EXECUTION_DIR}/SOURCE_FILES.txt")" -eq 1
test "$(grep -Fxc 'tests/__init__.py' "${EXECUTION_DIR}/SOURCE_FILES.txt" || true)" -eq 0

export SOURCE_ROOT EXECUTION_DIR
"${PYTHON}" - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["SOURCE_ROOT"])
execution = Path(os.environ["EXECUTION_DIR"])

tests_init = root / "tests" / "__init__.py"
tests_init.write_text(
    '"""Project test package for targeted module-name execution."""\n',
    encoding="utf-8",
    newline="\n",
)

source_files = execution / "SOURCE_FILES.txt"
text = source_files.read_text(encoding="utf-8")
anchor = "tests/test_planned_corruption.py\n"
if text.count(anchor) != 1 or "tests/__init__.py" in text:
    raise SystemExit("B0-v1 source-list identity mismatch")
source_files.write_text(
    text.replace(anchor, "tests/__init__.py\n" + anchor, 1),
    encoding="utf-8",
    newline="\n",
)

old_run = "20260809_h1_dlm_state_panels_b0_v1"
new_run = "20260809_h1_dlm_state_panels_b0_v2"
old_identity = "h1_dlm_state_panels_b0_v1"
new_identity = "h1_dlm_state_panels_b0_v2"

config = execution / "CONFIG.json"
text = config.read_text(encoding="utf-8")
config_schema = '"schema": "evidence_first_dlm_state_panels_v1"'
if (
    text.count(old_run) != 1
    or text.count(old_identity) != 2
    or text.count(config_schema) != 1
):
    raise SystemExit("B0-v1 config identity mismatch")
text = text.replace(old_run, new_run, 1)
text = text.replace(old_identity, new_identity, 1)
config.write_text(text, encoding="utf-8", newline="\n")

prepare = execution / "prepare_and_submit_once.sh"
text = prepare.read_text(encoding="utf-8")
submission_schema = '"schema": "evidence_first_dlm_state_panel_submission_v1"'
if (
    text.count(old_run) != 1
    or text.count(old_identity) != 2
    or text.count(submission_schema) != 1
):
    raise SystemExit("B0-v1 prepare identity mismatch")
text = text.replace(old_run, new_run, 1)
text = text.replace(old_identity, new_identity, 1)
prepare.write_text(text, encoding="utf-8", newline="\n")

sbatch = execution / "state_panels.sbatch"
text = sbatch.read_text(encoding="utf-8")
if text.count(old_run) != 3 or text.count("#SBATCH --job-name=h1-dlm-panels-b0") != 1:
    raise SystemExit("B0-v1 sbatch identity mismatch")
text = text.replace(old_run, new_run)
text = text.replace(
    "#SBATCH --job-name=h1-dlm-panels-b0",
    "#SBATCH --job-name=h1-dlm-panels-b0-v2",
    1,
)
sbatch.write_text(text, encoding="utf-8", newline="\n")

for path in (config, prepare, sbatch):
    current = path.read_text(encoding="utf-8")
    if old_run in current or old_identity in current:
        raise SystemExit(f"stale B0-v1 identity remains: {path}")
if config.read_text(encoding="utf-8").count(config_schema) != 1:
    raise SystemExit("B0 config schema changed")
if prepare.read_text(encoding="utf-8").count(submission_schema) != 1:
    raise SystemExit("B0 submission schema changed")
PY

test "$(sha256sum "${SOURCE_ROOT}/tests/__init__.py" | cut -d' ' -f1)" = \
  "${EXPECTED_TESTS_INIT_SHA256}"
bash -n "${EXECUTION_DIR}/prepare_and_submit_once.sh"
bash -n "${EXECUTION_DIR}/state_panels.sbatch"
grep -Fq '"run_root": "'"${V2_RUN_ROOT}"'"' "${EXECUTION_DIR}/CONFIG.json"
grep -Fq '#SBATCH --partition=gpu' "${EXECUTION_DIR}/state_panels.sbatch"
if grep -Fq 'gpu_long' "${EXECUTION_DIR}/state_panels.sbatch"; then
  echo 'forbidden gpu_long partition in B0-v2' >&2
  exit 3
fi

tar --sort=name --mtime='UTC 2026-08-09' --owner=0 --group=0 --numeric-owner \
  -czf "${V2_ARCHIVE}" -C "${SOURCE_ROOT}" .
V2_ARCHIVE_SHA256="$(sha256sum "${V2_ARCHIVE}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/B0_V2_REPAIR_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_state_panels_b0_v2_test_package_repair",
  "status": "pass",
  "source_archive_sha256": "${EXPECTED_SOURCE_ARCHIVE_SHA256}",
  "failed_v1_focused_test_sha256": "${EXPECTED_V1_FOCUSED_TEST_SHA256}",
  "failed_v1_reason": "tests_package_marker_omitted_from_frozen_archive",
  "tests_init_sha256": "${EXPECTED_TESTS_INIT_SHA256}",
  "v2_archive_sha256": "${V2_ARCHIVE_SHA256}",
  "scientific_algorithm_or_config_values_changed": false,
  "identity_metadata_changed": true,
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
    "$(basename "${V2_ARCHIVE}")" \
    B0_V2_REPAIR_RECORD.json
) > "${STAGE_ROOT}/B0_V2_STAGE_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c B0_V2_STAGE_SHA256.txt)
chmod 400 \
  "${STAGE_ROOT}/$(basename "${SELF}")" \
  "${V2_ARCHIVE}" \
  "${STAGE_ROOT}/B0_V2_REPAIR_RECORD.json" \
  "${STAGE_ROOT}/B0_V2_STAGE_SHA256.txt"

bash "${EXECUTION_DIR}/prepare_and_submit_once.sh" "${V2_ARCHIVE}"

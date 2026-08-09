#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V19_SOURCE="${PROJECT_ROOT}/runs/prepare_and_submit_v19_p0_legacy_identity_gate_repair_48b719c.sh"
V19_PARTIAL_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v18_raw256_assembly_p0_legacy_identity_gate_repair_v19"
V20_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v18_raw256_assembly_p0_legacy_identity_gate_repair_v20"
EXPECTED_SELF_SHA256="${1:?expected V20 generator SHA256}"
EXPECTED_V19_SOURCE_SHA256=a48858125e8e70f6febc6244dd7aa37636a21955b2d509d8d40a11e8dcc5c347
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v20_launcher_provenance_guard_generator_${EXPECTED_SELF_SHA256}"
GENERATED="${STAGE_ROOT}/prepare_and_submit_v20_launcher_provenance_guard_repair_generated.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V19_SOURCE}"
test "$(sha256sum "${V19_SOURCE}" | cut -d' ' -f1)" = \
  "${EXPECTED_V19_SOURCE_SHA256}"
test -d "${V19_PARTIAL_ROOT}"
test -d "${V19_PARTIAL_ROOT}/launchers"
test -z "$(find "${V19_PARTIAL_ROOT}/launchers" -mindepth 1 -maxdepth 1 -print -quit)"
test -f "${V19_PARTIAL_ROOT}/source/SOURCE_SHA256.txt"
test ! -e "${V19_PARTIAL_ROOT}/status/preparation_SUCCESS"
test ! -e "${V19_PARTIAL_ROOT}/V19_REPAIR_RECORD.json"
test ! -e "${V19_PARTIAL_ROOT}/V19_SHA256.txt"
test ! -e "${V19_PARTIAL_ROOT}/status/submitted_assemble256_job_id.txt"
test ! -e "${V20_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
export V19_SOURCE GENERATED
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V19_SOURCE"])
generated = Path(os.environ["GENERATED"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen V19 preparation script")

old_guard = '    if old_name in text or "_v18" in text:\n'
new_guard = (
    '    stale_launcher_markers = (\n'
    '        "submit_assemble256_v18",\n'
    '        "assemble256_v18",\n'
    '        "assemble_stage_v18",\n'
    '    )\n'
    '    if old_name in text or any(\n'
    '        marker in text for marker in stale_launcher_markers\n'
    '    ):\n'
)
if text.count(old_guard) != 1:
    raise SystemExit("V19 launcher guard identity mismatch")

text = text.replace("V19", "V20").replace("v19", "v20")
text = text.replace(old_guard, new_guard, 1)
if old_guard in text:
    raise SystemExit("overbroad V19 launcher guard remains")
if text.count("20260809_h1_chemistry_first_sft_v2_v18_raw256_assembly_p0_legacy_identity_gate_repair_v20") != 1:
    raise SystemExit("V20 run-root identity mismatch")
if "submit_assemble256_v20_once.sh" not in text:
    raise SystemExit("V20 submit launcher identity missing")
if '"schema": "h1_chemistry_first_raw256_assembly_p0_legacy_identity_gate_repair_v20"' not in text:
    raise SystemExit("V20 repair schema identity missing")
generated.write_text(text, encoding="utf-8", newline="\n")
PY

bash -n "${GENERATED}"
grep -Fq 'marker in text for marker in stale_launcher_markers' "${GENERATED}"
grep -Fq 'submit_assemble256_v20_once.sh' "${GENERATED}"
grep -Fq '#SBATCH --partition=normal' "${GENERATED}"

GENERATED_SHA256="$(sha256sum "${GENERATED}" | cut -d' ' -f1)"
cat > "${STAGE_ROOT}/V20_GENERATION_RECORD.json" <<EOF
{
  "schema": "h1_chemistry_first_v20_launcher_provenance_guard_generator",
  "status": "pass",
  "failed_v19_source_sha256": "${EXPECTED_V19_SOURCE_SHA256}",
  "failed_v19_reason": "launcher_guard_matched_v18_provenance_in_new_run_root",
  "generated_v20_sha256": "${GENERATED_SHA256}",
  "repair_scope": "replace_overbroad_v18_substring_guard_with_exact_old_root_and_launcher_markers",
  "p0_legacy_identity_gate_repair_changed": false,
  "raw_ledger_model_training_and_scientific_metrics_changed": false,
  "smact4_executed_on_a800": false,
  "broad_tests_repeated": false
}
EOF
(
  cd "${STAGE_ROOT}"
  sha256sum \
    "$(basename "${SELF}")" \
    "$(basename "${GENERATED}")" \
    V20_GENERATION_RECORD.json
) > "${STAGE_ROOT}/V20_GENERATOR_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c V20_GENERATOR_SHA256.txt)
chmod 400 "${STAGE_ROOT}"/*

bash "${GENERATED}" "${GENERATED_SHA256}"

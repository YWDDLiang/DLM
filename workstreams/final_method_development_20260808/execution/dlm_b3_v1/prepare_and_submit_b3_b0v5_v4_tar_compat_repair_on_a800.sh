#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V3_ADAPTER="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v3_generator_de2124f15f6777c126c0fa14c0d3ddf48663d7c68a619362203cee074b538d31/prepare_and_submit_b3_b0v5_v3_on_a800.sh"
EXPECTED_V3_ADAPTER_SHA256=51317582e78fca29c5989b28b3f4ebbe73a8111b844f43c918a07d53f8067a55
V3_STAGE="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v3_adapter_${EXPECTED_V3_ADAPTER_SHA256}"
V3_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_b0v5_v3"
V4_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_b0v5_v4"
EXPECTED_SELF_SHA256="${1:?expected B3-v4 tar-compat repair SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
GEN_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v4_generator_${EXPECTED_SELF_SHA256}"
GENERATED_ADAPTER="${GEN_ROOT}/prepare_and_submit_b3_b0v5_v4_on_a800.sh"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V3_ADAPTER}"
test "$(sha256sum "${V3_ADAPTER}" | cut -d' ' -f1)" = \
  "${EXPECTED_V3_ADAPTER_SHA256}"

# V3 passed all source identities and stopped before archive creation/SBatch
# because this A800 login node's tar lacks GNU reproducibility flags.
test -d "${V3_STAGE}/source"
test -f "${V3_STAGE}/SOURCE_FILES_V3.sha256"
test "$(sha256sum "${V3_STAGE}/SOURCE_FILES_V3.sha256" | cut -d' ' -f1)" = \
  ed3e5e16d0d91171142c0ab88a01e84ef1cefbc05e6056440af59b7c82653db0
test ! -e "${V3_STAGE}/dlm_b3_b0v5_v3_source.tar.gz"
test ! -e "${V3_STAGE}/B3_B0V5_V3_ADAPTATION_RECORD.json"
test ! -e "${V3_STAGE}/B3_B0V5_V3_SHA256.txt"
test ! -e "${V3_RUN_ROOT}"
test ! -e "${V4_RUN_ROOT}"
test ! -e "${GEN_ROOT}"

mkdir -p "${GEN_ROOT}"
export V3_ADAPTER GENERATED_ADAPTER
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V3_ADAPTER"])
target = Path(os.environ["GENERATED_ADAPTER"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen B3-v3 adapter")

old_tar = '''tar \\
  --sort=name \\
  --mtime='UTC 1970-01-01' \\
  --owner=0 --group=0 --numeric-owner \\
  -czf "${V3_ARCHIVE}" \\
  -C "${SOURCE_STAGE}" .'''
new_tar = '''tar -czf "${V3_ARCHIVE}" -C "${SOURCE_STAGE}" .'''
if text.count(old_tar) != 1:
    raise SystemExit("B3-v3 GNU tar block identity mismatch")
text = text.replace(old_tar, new_tar, 1)
text = text.replace("V3", "V4").replace("v3", "v4")

for stale in (
    "V3_RUN_ROOT",
    "B3_B0V5_V3",
    "b0v5_v3",
    "b3_b0v5_v3",
    "--sort=name",
    "--mtime='UTC 1970-01-01'",
    "--numeric-owner",
):
    if stale in text:
        raise SystemExit(f"stale V3/tar identity remains: {stale}")
expected_tar = 'tar -czf "${V4_ARCHIVE}" -C "${SOURCE_STAGE}" .'
if text.count(expected_tar) != 1:
    raise SystemExit("B3-v4 portable tar identity mismatch")
if text.count("(old_root, new_root, 3)") != 2:
    raise SystemExit("B3-v4 sbatch rewrite-count identity mismatch")

target.write_text(text, encoding="utf-8", newline="\n")
PY

chmod 500 "${GENERATED_ADAPTER}"
bash -n "${GENERATED_ADAPTER}"
GENERATED_SHA256="$(sha256sum "${GENERATED_ADAPTER}" | cut -d' ' -f1)"

cat > "${GEN_ROOT}/B3_B0V5_V4_GENERATION_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_b3_b0v5_v4_tar_compat_repair",
  "status": "pass",
  "frozen_v3_adapter_sha256": "${EXPECTED_V3_ADAPTER_SHA256}",
  "frozen_v3_source_files_sha256": "ed3e5e16d0d91171142c0ab88a01e84ef1cefbc05e6056440af59b7c82653db0",
  "generated_v4_adapter_sha256": "${GENERATED_SHA256}",
  "repair_scope": "unsupported_gnu_tar_reproducibility_flags_to_portable_tar_czf",
  "training_contract_changed": false,
  "scientific_contract_changed": false,
  "automatic_body64_submission": false,
  "automatic_ratio_sweep": false,
  "automatic_downstream": false,
  "automatic_sun": false,
  "automatic_rl": false
}
EOF
(
  cd "${GEN_ROOT}"
  sha256sum \
    "$(basename "${GENERATED_ADAPTER}")" \
    B3_B0V5_V4_GENERATION_RECORD.json
) > "${GEN_ROOT}/B3_B0V5_V4_GENERATION_SHA256.txt"
(cd "${GEN_ROOT}" && sha256sum -c B3_B0V5_V4_GENERATION_SHA256.txt)
chmod 400 \
  "${GEN_ROOT}/B3_B0V5_V4_GENERATION_RECORD.json" \
  "${GEN_ROOT}/B3_B0V5_V4_GENERATION_SHA256.txt"

bash "${GENERATED_ADAPTER}" "${GENERATED_SHA256}"

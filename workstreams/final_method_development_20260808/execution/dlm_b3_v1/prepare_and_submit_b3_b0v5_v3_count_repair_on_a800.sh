#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V2_ADAPTER="${PROJECT_ROOT}/runs/transfer_prepare_and_submit_b3_b0v5_v2_642aa22.sh"
EXPECTED_V2_ADAPTER_SHA256=0eca13cb57bc0f15cbcd00b2d4de744e9bd3ec22398997d744bb9d4930a1d82e
V2_STAGE="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v2_adapter_${EXPECTED_V2_ADAPTER_SHA256}"
V2_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_b0v5_v2"
V3_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_b0v5_v3"
EXPECTED_SELF_SHA256="${1:?expected B3-v3 count-repair SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
GEN_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v3_generator_${EXPECTED_SELF_SHA256}"
GENERATED_ADAPTER="${GEN_ROOT}/prepare_and_submit_b3_b0v5_v3_on_a800.sh"

test -f "${SELF}"
test -x "${PYTHON}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${V2_ADAPTER}"
test "$(sha256sum "${V2_ADAPTER}" | cut -d' ' -f1)" = \
  "${EXPECTED_V2_ADAPTER_SHA256}"

# Frozen V2 failed before archive creation or sbatch because the old run-root
# appears three times in each sbatch file, not once. Preserve that evidence and
# derive V3 only from the byte-frozen V2 adapter.
test -d "${V2_STAGE}/source"
test ! -e "${V2_STAGE}/dlm_b3_b0v5_v2_source.tar.gz"
test ! -e "${V2_STAGE}/B3_B0V5_V2_ADAPTATION_RECORD.json"
test ! -e "${V2_STAGE}/B3_B0V5_V2_SHA256.txt"
test ! -e "${V2_RUN_ROOT}"
test ! -e "${V3_RUN_ROOT}"
test ! -e "${GEN_ROOT}"

mkdir -p "${GEN_ROOT}"
export V2_ADAPTER GENERATED_ADAPTER
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["V2_ADAPTER"])
target = Path(os.environ["GENERATED_ADAPTER"])
text = source.read_text(encoding="utf-8")
if "\r" in text:
    raise SystemExit("CR byte in frozen B3-v2 adapter")

train_old = '''rewrite(
    "train_b3.sbatch",
    [
        (old_root, new_root, 1),'''
train_new = '''rewrite(
    "train_b3.sbatch",
    [
        (old_root, new_root, 3),'''
score_old = '''rewrite(
    "score_b3_panels.sbatch",
    [
        (old_root, new_root, 1),'''
score_new = '''rewrite(
    "score_b3_panels.sbatch",
    [
        (old_root, new_root, 3),'''
for old in (train_old, score_old):
    if text.count(old) != 1:
        raise SystemExit("B3-v2 rewrite-count anchor mismatch")
text = text.replace(train_old, train_new, 1)
text = text.replace(score_old, score_new, 1)
text = text.replace("V2", "V3").replace("v2", "v3")

for stale in (
    "V2_RUN_ROOT",
    "B3_B0V5_V2",
    "b0v5_v2",
    "b3_b0v5_v2",
):
    if stale in text:
        raise SystemExit(f"stale V2 identity remains: {stale}")
if text.count("(old_root, new_root, 3)") != 2:
    raise SystemExit("B3-v3 sbatch rewrite-count identity mismatch")

target.write_text(text, encoding="utf-8", newline="\n")
PY

chmod 500 "${GENERATED_ADAPTER}"
bash -n "${GENERATED_ADAPTER}"
GENERATED_SHA256="$(sha256sum "${GENERATED_ADAPTER}" | cut -d' ' -f1)"

cat > "${GEN_ROOT}/B3_B0V5_V3_GENERATION_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_b3_b0v5_v3_count_repair",
  "status": "pass",
  "frozen_v2_adapter_sha256": "${EXPECTED_V2_ADAPTER_SHA256}",
  "generated_v3_adapter_sha256": "${GENERATED_SHA256}",
  "repair_scope": "train_and_score_old_run_root_expected_count_1_to_3",
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
    B3_B0V5_V3_GENERATION_RECORD.json
) > "${GEN_ROOT}/B3_B0V5_V3_GENERATION_SHA256.txt"
(cd "${GEN_ROOT}" && sha256sum -c B3_B0V5_V3_GENERATION_SHA256.txt)
chmod 400 \
  "${GEN_ROOT}/B3_B0V5_V3_GENERATION_RECORD.json" \
  "${GEN_ROOT}/B3_B0V5_V3_GENERATION_SHA256.txt"

bash "${GENERATED_ADAPTER}" "${GENERATED_SHA256}"

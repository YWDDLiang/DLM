#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
ORIGINAL_ARCHIVE="${PROJECT_ROOT}/runs/transfer_dlm_b3_v1_52d677e.tar.gz"
B0_V5_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_state_panels_b0_v5"
B0_V5_SOURCE="${B0_V5_ROOT}/source"
B0_V5_PANELS="${B0_V5_ROOT}/panels"
V2_RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_dlm_b3_safe_axis_2to1_b0v5_v2"
EXPECTED_SELF_SHA256="${1:?expected B3-v2 adapter SHA256}"
EXPECTED_ORIGINAL_ARCHIVE_SHA256=d54a07ddfee2f361d067b5975747d28bb8d4b656210cbedcc6f1c3e97a7e887e
EXPECTED_PANEL_MANIFEST_SHA256=6cc3d81074a3e472b39c93090d3d4a85c6565d92eb7eb3c5c18a57cf9f966937
EXPECTED_PANEL_TERMINAL_SHA256=18c51f6e86bb1bfe666b8dfb531b23cee5338d979729db84540d49e53ee5758c
EXPECTED_PANEL_CONFIG_SHA256=6051b1f818d878df59c54b58b90c5acd0979b4cc4ea5bfb765c6a520131aa2e7
EXPECTED_PANEL_EVALUATOR_SHA256=e2cbc6d55ba570a7452b3384d173239b81766c1bd15c56168cb330a0979de0c4
EXPECTED_PANEL_SCORER_SHA256=df279c21a125aaba68f226664e88f998b8a6f863386bda854e491305098061e9
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_dlm_b3_b0v5_v2_adapter_${EXPECTED_SELF_SHA256}"
SOURCE_STAGE="${STAGE_ROOT}/source"
V2_ARCHIVE="${STAGE_ROOT}/dlm_b3_b0v5_v2_source.tar.gz"
B3_REL=workstreams/final_method_development_20260808/execution/dlm_b3_v1
PANEL_REL=workstreams/final_method_development_20260808/execution/dlm_state_panels_v1

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -f "${ORIGINAL_ARCHIVE}"
test "$(sha256sum "${ORIGINAL_ARCHIVE}" | cut -d' ' -f1)" = \
  "${EXPECTED_ORIGINAL_ARCHIVE_SHA256}"
test -f "${B0_V5_ROOT}/status/job_31323_SUCCESS"
test ! -e "${B0_V5_ROOT}/status/job_31323_FAILED"
test -f "${B0_V5_PANELS}/_SUCCESS"
test "$(sha256sum "${B0_V5_PANELS}/state_panel_manifest.json" | cut -d' ' -f1)" = \
  "${EXPECTED_PANEL_MANIFEST_SHA256}"
test "$(sha256sum "${B0_V5_PANELS}/terminal_report.json" | cut -d' ' -f1)" = \
  "${EXPECTED_PANEL_TERMINAL_SHA256}"
test "$(sha256sum "${B0_V5_SOURCE}/${PANEL_REL}/CONFIG.json" | cut -d' ' -f1)" = \
  "${EXPECTED_PANEL_CONFIG_SHA256}"
test "$(sha256sum "${B0_V5_SOURCE}/${PANEL_REL}/evaluate_state_panels.py" | cut -d' ' -f1)" = \
  "${EXPECTED_PANEL_EVALUATOR_SHA256}"
test "$(sha256sum "${B0_V5_SOURCE}/${PANEL_REL}/score_frozen_state_panels.py" | cut -d' ' -f1)" = \
  "${EXPECTED_PANEL_SCORER_SHA256}"
test ! -e "${V2_RUN_ROOT}"
test ! -e "${STAGE_ROOT}"

mkdir -p "${SOURCE_STAGE}"
cp "${SELF}" "${STAGE_ROOT}/"
tar -xzf "${ORIGINAL_ARCHIVE}" -C "${SOURCE_STAGE}"
test -f "${SOURCE_STAGE}/${B3_REL}/SOURCE_FILES.txt"

cp "${B0_V5_SOURCE}/${PANEL_REL}/CONFIG.json" \
  "${SOURCE_STAGE}/${PANEL_REL}/CONFIG.json"
cp "${B0_V5_SOURCE}/${PANEL_REL}/evaluate_state_panels.py" \
  "${SOURCE_STAGE}/${PANEL_REL}/evaluate_state_panels.py"
cp "${B0_V5_SOURCE}/${PANEL_REL}/score_frozen_state_panels.py" \
  "${SOURCE_STAGE}/${PANEL_REL}/score_frozen_state_panels.py"

export SOURCE_STAGE B3_REL V2_RUN_ROOT EXPECTED_PANEL_MANIFEST_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["SOURCE_STAGE"])
b3 = source / os.environ["B3_REL"]
new_root = Path(os.environ["V2_RUN_ROOT"]).name
panel_sha = os.environ["EXPECTED_PANEL_MANIFEST_SHA256"]
old_root = "20260809_h1_dlm_b3_safe_axis_2to1_v1"
old_panel = "20260809_h1_dlm_state_panels_b0_v1"
new_panel = "20260809_h1_dlm_state_panels_b0_v5"

def rewrite(relative, replacements):
    path = b3 / relative
    text = path.read_text(encoding="utf-8")
    if "\r" in text:
        raise SystemExit(f"CR byte in {relative}")
    for old, new, count in replacements:
        if text.count(old) != count:
            raise SystemExit(f"identity mismatch in {relative}: {old}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")

prepare = b3 / "prepare_and_submit_once.sh"
text = prepare.read_text(encoding="utf-8")
if text.count(old_root) != 1 or text.count(old_panel) != 1:
    raise SystemExit("B3-v1 prepare path identity mismatch")
if text.count('"identity": "h1_dlm_b3_safe_axis_2to1_v1",') != 1:
    raise SystemExit("B3-v1 submission identity mismatch")
text = text.replace(old_root, new_root)
text = text.replace(old_panel, new_panel)
text = text.replace(
    '"identity": "h1_dlm_b3_safe_axis_2to1_v1",',
    '"identity": "h1_dlm_b3_safe_axis_2to1_b0v5_v2",',
)
terminal_anchor = '"${PYTHON}" - "${PANEL_ROOT}/terminal_report.json" <<\'PY\'\n'
if text.count(terminal_anchor) != 1:
    raise SystemExit("B3-v1 terminal audit anchor mismatch")
text = text.replace(
    terminal_anchor,
    f'test "$(sha256sum "${{PANEL_ROOT}}/state_panel_manifest.json" | cut -d\' \' -f1)" = "{panel_sha}"\n'
    + terminal_anchor,
    1,
)
old_terminal_gate = (
    '    or report.get("automatic_b3_submission") is not False\n'
)
new_terminal_gate = (
    old_terminal_gate
    + f'    or report.get("state_panel_manifest_sha256") != "{panel_sha}"\n'
    + '    or report.get("scientific_score_batch_size") != 1\n'
    + '    or report.get("producer_rescore_audit", {}).get("status") != "pass"\n'
    + '    or report.get("producer_rescore_audit", {}).get("max_abs_delta") != 0.0\n'
)
if text.count(old_terminal_gate) != 1:
    raise SystemExit("B3-v1 terminal gate identity mismatch")
text = text.replace(old_terminal_gate, new_terminal_gate, 1)
record_anchor = '    "frozen_b0_state_panel_manifest_sha256": sys.argv[6],\n'
if text.count(record_anchor) != 1:
    raise SystemExit("B3-v1 submission record anchor mismatch")
text = text.replace(
    record_anchor,
    record_anchor
    + '    "frozen_b0_state_panel_identity": "h1_dlm_state_panels_b0_v5",\n'
    + '    "scientific_score_batch_size": 1,\n',
    1,
)
prepare.write_text(text, encoding="utf-8", newline="\n")

rewrite(
    "CONFIG.json",
    [
        (old_root, new_root, 1),
        (old_panel, new_panel, 1),
        (
            "h1_dlm_b3_safe_axis_2to1_v1",
            "h1_dlm_b3_safe_axis_2to1_b0v5_v2",
            1,
        ),
    ],
)
rewrite(
    "train_b3.sbatch",
    [
        (old_root, new_root, 1),
        ("#SBATCH --job-name=h1-dlm-b3", "#SBATCH --job-name=h1-dlm-b3-b0v5-v2", 1),
    ],
)
rewrite(
    "score_b3_panels.sbatch",
    [
        (old_root, new_root, 1),
        (old_panel, new_panel, 1),
        (
            "#SBATCH --job-name=h1-dlm-b3-panels",
            "#SBATCH --job-name=h1-dlm-b3-b0v5-v2-panels",
            1,
        ),
    ],
)

for relative in (
    "prepare_and_submit_once.sh",
    "CONFIG.json",
    "train_b3.sbatch",
    "score_b3_panels.sbatch",
):
    text = (b3 / relative).read_text(encoding="utf-8")
    if old_root in text or old_panel in text:
        raise SystemExit(f"stale B3-v1 path remains in {relative}")
PY

bash -n "${SOURCE_STAGE}/${B3_REL}/prepare_and_submit_once.sh"
bash -n "${SOURCE_STAGE}/${B3_REL}/train_b3.sbatch"
bash -n "${SOURCE_STAGE}/${B3_REL}/score_b3_panels.sbatch"
grep -Fq '#SBATCH --partition=gpu' "${SOURCE_STAGE}/${B3_REL}/train_b3.sbatch"
grep -Fq '#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2' \
  "${SOURCE_STAGE}/${B3_REL}/train_b3.sbatch"
grep -Fq -- '--max-train-steps 1696' "${SOURCE_STAGE}/${B3_REL}/train_b3.sbatch"
grep -Fq -- '--lr 5e-5' "${SOURCE_STAGE}/${B3_REL}/train_b3.sbatch"
grep -Fq '"score_batch_size": 1' "${SOURCE_STAGE}/${PANEL_REL}/CONFIG.json"
if grep -R -Fq 'gpu_long' "${SOURCE_STAGE}/${B3_REL}"; then
  echo "B3-v2 must not use gpu_long" >&2
  exit 3
fi

(
  cd "${SOURCE_STAGE}"
  while IFS= read -r relative; do
    test -n "${relative}" || continue
    test -f "${relative}"
    sha256sum "${relative}"
  done < "${B3_REL}/SOURCE_FILES.txt"
) > "${STAGE_ROOT}/SOURCE_FILES_V2.sha256"
(cd "${SOURCE_STAGE}" && sha256sum -c "${STAGE_ROOT}/SOURCE_FILES_V2.sha256")

tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 --numeric-owner \
  -czf "${V2_ARCHIVE}" \
  -C "${SOURCE_STAGE}" .
V2_ARCHIVE_SHA256="$(sha256sum "${V2_ARCHIVE}" | cut -d' ' -f1)"
SOURCE_FILES_SHA256="$(sha256sum "${STAGE_ROOT}/SOURCE_FILES_V2.sha256" | cut -d' ' -f1)"

cat > "${STAGE_ROOT}/B3_B0V5_V2_ADAPTATION_RECORD.json" <<EOF
{
  "schema": "evidence_first_dlm_b3_b0v5_v2_adaptation",
  "status": "pass",
  "source_commit_v1": "52d677e",
  "source_archive_v1_sha256": "${EXPECTED_ORIGINAL_ARCHIVE_SHA256}",
  "b0_v5_panel_manifest_sha256": "${EXPECTED_PANEL_MANIFEST_SHA256}",
  "b0_v5_panel_terminal_sha256": "${EXPECTED_PANEL_TERMINAL_SHA256}",
  "b0_v5_panel_source_commit": "2803439",
  "source_files_v2_sha256": "${SOURCE_FILES_SHA256}",
  "source_archive_v2_sha256": "${V2_ARCHIVE_SHA256}",
  "training_contract_changed": false,
  "training_updates": 1696,
  "training_learning_rate": 0.00005,
  "training_partition": "gpu",
  "training_a800": 2,
  "scoring_batch_size": 1,
  "automatic_body64_submission": false,
  "automatic_ratio_sweep": false,
  "automatic_downstream": false,
  "automatic_sun": false,
  "automatic_rl": false
}
EOF
(
  cd "${STAGE_ROOT}"
  sha256sum \
    "$(basename "${SELF}")" \
    "$(basename "${V2_ARCHIVE}")" \
    SOURCE_FILES_V2.sha256 \
    B3_B0V5_V2_ADAPTATION_RECORD.json
) > "${STAGE_ROOT}/B3_B0V5_V2_SHA256.txt"
(cd "${STAGE_ROOT}" && sha256sum -c B3_B0V5_V2_SHA256.txt)
chmod 400 \
  "${STAGE_ROOT}/$(basename "${SELF}")" \
  "${V2_ARCHIVE}" \
  "${STAGE_ROOT}/SOURCE_FILES_V2.sha256" \
  "${STAGE_ROOT}/B3_B0V5_V2_ADAPTATION_RECORD.json" \
  "${STAGE_ROOT}/B3_B0V5_V2_SHA256.txt"

bash "${SOURCE_STAGE}/${B3_REL}/prepare_and_submit_once.sh" "${V2_ARCHIVE}"

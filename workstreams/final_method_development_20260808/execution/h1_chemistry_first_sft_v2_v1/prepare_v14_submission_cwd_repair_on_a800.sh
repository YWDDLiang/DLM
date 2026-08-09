#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_V10="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
PARENT_V13="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13"
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_submission_cwd_repair_v14"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SELF_SHA256="${1:?expected preparation-script SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
EXPECTED_V13_PREPARATION_SHA256=a5ed95ffc02ac2eadb8719ad0b9bd58350c963cb5178cff15ecca5c9581efa25
EXPECTED_V13_MANIFEST_SHA256=cf56f289f72647aa4157480c21116b664390c1b28ea076f37a29fa65de710bb0
EXPECTED_AUTHORIZATION_SHA256=90f0bad19d16322122be7516243023fe8e97bf90c4c4707327e981abd2d56a6a
V13_NAME=20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13
V14_NAME=20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_submission_cwd_repair_v14

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test ! -e "${RUN_ROOT}"
test -f "${PARENT_V13}/status/preparation_SUCCESS"
test "$(sha256sum "${PARENT_V13}/V13_PREPARATION_RECORD.json" | cut -d' ' -f1)" = \
  "${EXPECTED_V13_PREPARATION_SHA256}"
test "$(sha256sum "${PARENT_V13}/V13_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V13_MANIFEST_SHA256}"
test "$(sha256sum "${PARENT_V13}/AUTHORIZATION.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUTHORIZATION_SHA256}"
test ! -e "${PARENT_V13}/submission_record.json"
test ! -e "${PARENT_V13}/planner256"
test ! -e "${PARENT_V13}/.submit_v13_lock"

mkdir -p "${RUN_ROOT}/launchers" "${RUN_ROOT}/logs" "${RUN_ROOT}/preflight" \
  "${RUN_ROOT}/status"
ln -s "${PARENT_V10}/source" "${RUN_ROOT}/source"
ln -s "${PARENT_V10}/source_archive.tar.gz" "${RUN_ROOT}/source_archive.tar.gz"
ln -s "${PARENT_V10}/training" "${RUN_ROOT}/training"
cp "${PARENT_V13}/AUTHORIZATION.json" "${RUN_ROOT}/AUTHORIZATION.json"
cp "${PARENT_V13}/LEDGER256.json" "${RUN_ROOT}/LEDGER256.json"
cp "${SELF}" "${RUN_ROOT}/prepare_v14_submission_cwd_repair_on_a800.sh"

export PARENT_V13 RUN_ROOT V13_NAME V14_NAME
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

parent = Path(os.environ["PARENT_V13"])
root = Path(os.environ["RUN_ROOT"])
old = os.environ["V13_NAME"]
new = os.environ["V14_NAME"]

planner = (parent / "launchers/planner256_v13.sbatch").read_text(encoding="utf-8")
submit = (parent / "launchers/submit_v13_once.sh").read_text(encoding="utf-8")
if planner.count(old) < 1 or submit.count(old) < 1:
    raise SystemExit("V13 run identity missing from launcher")
planner = planner.replace(old, new).replace("h1-cf-p256-v13", "h1-cf-p256-v14")
submit = submit.replace(old, new).replace(
    'sha256sum -c "${RUN_ROOT}/V13_SHA256.txt"',
    '(cd "${RUN_ROOT}" && sha256sum -c V14_SHA256.txt)',
)
submit = submit.replace("planner256_v13.sbatch", "planner256_v14.sbatch")
submit = submit.replace(".submit_v13_lock", ".submit_v14_lock")
submit = submit.replace(
    "h1_chemistry_first_v13_submission_record_v1",
    "h1_chemistry_first_v14_submission_record_v1",
)
if old in planner or old in submit:
    raise SystemExit("stale V13 run identity remains")
if '(cd "${RUN_ROOT}" && sha256sum -c V14_SHA256.txt)' not in submit:
    raise SystemExit("cwd repair was not applied exactly once")
(root / "launchers/planner256_v14.sbatch").write_text(
    planner, encoding="utf-8", newline="\n"
)
(root / "launchers/submit_v14_once.sh").write_text(
    submit, encoding="utf-8", newline="\n"
)
PY

chmod 500 "${RUN_ROOT}/launchers/planner256_v14.sbatch" \
  "${RUN_ROOT}/launchers/submit_v14_once.sh"
bash -n "${RUN_ROOT}/launchers/planner256_v14.sbatch"
bash -n "${RUN_ROOT}/launchers/submit_v14_once.sh"
grep -Fq '(cd "${RUN_ROOT}" && sha256sum -c V14_SHA256.txt)' \
  "${RUN_ROOT}/launchers/submit_v14_once.sh"

export EXPECTED_V13_PREPARATION_SHA256 EXPECTED_V13_MANIFEST_SHA256
export EXPECTED_AUTHORIZATION_SHA256
"${PYTHON}" - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "h1_chemistry_first_v14_preparation_record_v1",
    "status": "pass",
    "run_root": str(root),
    "parent_v13_root": os.environ["PARENT_V13"],
    "parent_v13_preparation_sha256": os.environ["EXPECTED_V13_PREPARATION_SHA256"],
    "parent_v13_manifest_sha256": os.environ["EXPECTED_V13_MANIFEST_SHA256"],
    "authorization_sha256": os.environ["EXPECTED_AUTHORIZATION_SHA256"],
    "repair_scope": "submission_manifest_working_directory_only",
    "failure_boundary": "before_lock_before_sbatch_before_generation",
    "parent_submission_record_absent": True,
    "parent_planner256_absent": True,
    "science_contract_changed": False,
    "sampling_contract_changed": False,
    "source_or_checkpoint_changed": False,
    "broad_tests_repeated": False,
    "smact4_executed_on_a800": False,
    "launcher_sha256": {
        path.name: sha(path) for path in sorted((root / "launchers").iterdir())
    },
}
(root / "V14_PREPARATION_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

tar -czf "${RUN_ROOT}/launcher_archive.tar.gz" -C "${RUN_ROOT}" \
  AUTHORIZATION.json LEDGER256.json launchers V14_PREPARATION_RECORD.json
(
  cd "${RUN_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum AUTHORIZATION.json LEDGER256.json V14_PREPARATION_RECORD.json \
    launcher_archive.tar.gz prepare_v14_submission_cwd_repair_on_a800.sh
) > "${RUN_ROOT}/V14_SHA256.txt"
(cd "${RUN_ROOT}" && sha256sum -c V14_SHA256.txt)
touch "${RUN_ROOT}/status/preparation_SUCCESS"
chmod 400 "${RUN_ROOT}/AUTHORIZATION.json" "${RUN_ROOT}/LEDGER256.json" \
  "${RUN_ROOT}/V14_PREPARATION_RECORD.json" "${RUN_ROOT}/V14_SHA256.txt" \
  "${RUN_ROOT}/launcher_archive.tar.gz" \
  "${RUN_ROOT}/prepare_v14_submission_cwd_repair_on_a800.sh"
cat "${RUN_ROOT}/V14_PREPARATION_RECORD.json"

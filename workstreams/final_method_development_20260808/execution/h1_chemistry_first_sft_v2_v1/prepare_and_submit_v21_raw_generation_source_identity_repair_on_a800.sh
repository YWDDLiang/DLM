#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V20_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v18_raw256_assembly_p0_legacy_identity_gate_repair_v20"
V21_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v20_raw256_assembly_raw_source_identity_repair_v21"
AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
EXPECTED_SELF_SHA256="${1:?expected V21 generator SHA256}"
EXPECTED_V20_SHA_FILE_SHA256=95fb3009ed7b49f68efe3573d09e7e45075b2c18b81008cdabfb3c51cca95626
EXPECTED_V20_SOURCE_INVENTORY_SHA256=319b75917ac5f09d73be84a1c8adf266948f27005ae9a561c693f4b5a62f19d2
EXPECTED_V20_SOURCE_ARCHIVE_SHA256=51aae64a6fbd1db17a374746b21373ac055298ce00d305a343c517cfd684f4fa
EXPECTED_V20_STDERR_SHA256=72d87816fb969a57bbd013a387300c3191eb2cee13afa22ceb70e358a8968477
EXPECTED_AUDIT_MANIFEST_SHA256=ed17201d01a5f4f3a601892309ad671b45fe55d41cd15b1252aac8053bf4c6c4
EXPECTED_RAW_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
SELF="$(realpath "${BASH_SOURCE[0]}")"
STAGE_ROOT="${PROJECT_ROOT}/runs/transfer_v21_raw_generation_source_identity_repair_${EXPECTED_SELF_SHA256}"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -d "${V20_ROOT}"
test ! -e "${V21_ROOT}"
test ! -e "${STAGE_ROOT}"
test "$(sha256sum "${V20_ROOT}/V20_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V20_SHA_FILE_SHA256}"
(cd "${V20_ROOT}" && sha256sum -c V20_SHA256.txt)
test "$(sha256sum "${V20_ROOT}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V20_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${V20_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_V20_SOURCE_ARCHIVE_SHA256}"
test "$(sha256sum "${V20_ROOT}/logs/31318_assemble256.err" | cut -d' ' -f1)" = \
  "${EXPECTED_V20_STDERR_SHA256}"
test "$(sha256sum "${AUDIT_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"
test -f "${AUDIT_ROOT}/_SUCCESS"

mkdir "${STAGE_ROOT}"
cp "${SELF}" "${STAGE_ROOT}/"
sacct -n -j 31318 --format=JobID,State,ExitCode,Elapsed,Partition,NodeList -P \
  > "${STAGE_ROOT}/sacct_v20_before_v21.txt"
test "$(awk -F'|' '$1 == "31318" {print $2 "|" $3 "|" $5}' \
  "${STAGE_ROOT}/sacct_v20_before_v21.txt")" = 'FAILED|2:0|normal'

export V20_ROOT AUDIT_ROOT EXPECTED_RAW_SOURCE_INVENTORY_SHA256
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

v20 = Path(os.environ["V20_ROOT"])
audit = Path(os.environ["AUDIT_ROOT"])
stage = json.loads(
    (v20 / "planner256/terminal/stage_summary.json").read_text(encoding="utf-8")
)
candidate = json.loads(
    (v20 / "planner256/terminal/sft_v2_terminal_report.json").read_text(
        encoding="utf-8"
    )
)
manifest = json.loads((audit / "MANIFEST.json").read_text(encoding="utf-8"))
if stage.get("status") != "engineering_failure":
    raise SystemExit("V20 stage status mismatch")
if candidate.get("reason") != "local_smact4_bundle_import_exit_1":
    raise SystemExit("V20 failure reason mismatch")
if (
    manifest.get("status") != "pass"
    or manifest.get("stage") != 256
    or manifest.get("denominator") != 256
    or manifest.get("source_inventory_sha256")
    != os.environ["EXPECTED_RAW_SOURCE_INVENTORY_SHA256"]
    or manifest.get("arms") != ["p0", "sft_v2"]
):
    raise SystemExit("frozen local SMACT4 manifest identity mismatch")
PY

mkdir -p \
  "${V21_ROOT}/launchers" \
  "${V21_ROOT}/logs" \
  "${V21_ROOT}/planner256" \
  "${V21_ROOT}/preflight" \
  "${V21_ROOT}/status"
cp -a "${V20_ROOT}/source" "${V21_ROOT}/source"
cp "${V20_ROOT}/source_archive.tar.gz" "${V21_ROOT}/source_archive.tar.gz"
cp "${V20_ROOT}/submission256_record.json" "${V21_ROOT}/submission256_record.json"
sha256sum "${V21_ROOT}/submission256_record.json" \
  > "${V21_ROOT}/submission256_record.sha256"
ln -s "$(realpath "${V20_ROOT}/training")" "${V21_ROOT}/training"
for arm in p0 sft_v2; do
  ln -s "$(realpath "${V20_ROOT}/planner256/${arm}")" \
    "${V21_ROOT}/planner256/${arm}"
done
for name in \
  planner256_p0_exit_code.txt planner256_p0_SUCCESS \
  planner256_sft_v2_exit_code.txt planner256_sft_v2_SUCCESS \
  submitted_planner256_job_id.txt; do
  cp "${V20_ROOT}/status/${name}" "${V21_ROOT}/status/${name}"
done

export V20_ROOT V21_ROOT EXPECTED_RAW_SOURCE_INVENTORY_SHA256
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

old_root = Path(os.environ["V20_ROOT"])
new_root = Path(os.environ["V21_ROOT"])
raw_source = os.environ["EXPECTED_RAW_SOURCE_INVENTORY_SHA256"]
old_name = old_root.name
new_name = new_root.name
mapping = {
    "submit_assemble256_v20_once.sh": "submit_assemble256_v21_once.sh",
    "assemble256_v20.sbatch": "assemble256_v21.sbatch",
    "assemble_stage_v20.sbatch": "assemble_stage_v21.sbatch",
}
for old_file, new_file in mapping.items():
    text = (old_root / "launchers" / old_file).read_text(encoding="utf-8")
    if "\r" in text or text.count(old_name) < 1:
        raise SystemExit(f"V20 launcher identity mismatch: {old_file}")
    text = text.replace(old_name, new_name)
    text = text.replace("submit_assemble256_v20", "submit_assemble256_v21")
    text = text.replace("assemble256_v20", "assemble256_v21")
    text = text.replace("assemble_stage_v20", "assemble_stage_v21")
    text = text.replace("h1-cf-a256-v20", "h1-cf-a256-v21")

    if old_file == "submit_assemble256_v20_once.sh":
        source_arg = (
            'EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected '
            'SOURCE_SHA256.txt digest}"\n'
        )
        if text.count(source_arg) != 1:
            raise SystemExit("V21 submit source argument identity mismatch")
        text = text.replace(
            source_arg,
            source_arg
            + f"EXPECTED_RAW_SOURCE_INVENTORY_SHA256={raw_source}\n",
            1,
        )
        audit_guard = 'test ! -e "${REPAIR_ROOT}/assembly256_submission_record.json"\n'
        if text.count(audit_guard) != 1:
            raise SystemExit("V21 audit guard insertion identity mismatch")
        manifest_check = """observed_raw_source="$("${LEGACY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_inventory_sha256"])' "${LOCAL_SMACT4_AUDIT_ROOT}/MANIFEST.json")"
test "${observed_raw_source}" = "${EXPECTED_RAW_SOURCE_INVENTORY_SHA256}"
"""
        text = text.replace(audit_guard, manifest_check + audit_guard, 1)
        export_anchor = (
            'common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256='
            '${EXPECTED_SOURCE_INVENTORY_SHA256},'
        )
        if text.count(export_anchor) != 1:
            raise SystemExit("V21 common export identity mismatch")
        text = text.replace(
            export_anchor,
            export_anchor
            + 'EXPECTED_RAW_SOURCE_INVENTORY_SHA256='
            '${EXPECTED_RAW_SOURCE_INVENTORY_SHA256},',
            1,
        )

    if old_file == "assemble_stage_v20.sbatch":
        source_guard = ': "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"\n'
        if text.count(source_guard) != 1:
            raise SystemExit("V21 stage source guard identity mismatch")
        text = text.replace(
            source_guard,
            source_guard
            + ': "${EXPECTED_RAW_SOURCE_INVENTORY_SHA256:?missing raw source identity}"\n',
            1,
        )
        old_arg = '--source-inventory-sha256 "${EXPECTED_SOURCE_INVENTORY_SHA256}" \\\n'
        new_arg = '--source-inventory-sha256 "${EXPECTED_RAW_SOURCE_INVENTORY_SHA256}" \\\n'
        if text.count(old_arg) != 1:
            raise SystemExit("V21 verifier source argument identity mismatch")
        text = text.replace(old_arg, new_arg, 1)

    if old_name in text or "v20" in text or "V20" in text:
        raise SystemExit(f"stale V20 launcher marker: {old_file}")
    (new_root / "launchers" / new_file).write_text(
        text, encoding="utf-8", newline="\n"
    )
PY

for path in "${V21_ROOT}"/launchers/*; do
  bash -n "${path}"
  grep -Fq "${V21_ROOT##*/}" "${path}"
done
grep -Fq 'EXPECTED_RAW_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60' \
  "${V21_ROOT}/launchers/submit_assemble256_v21_once.sh"
grep -Fq -- '--source-inventory-sha256 "${EXPECTED_RAW_SOURCE_INVENTORY_SHA256}"' \
  "${V21_ROOT}/launchers/assemble_stage_v21.sbatch"
grep -Fq '#SBATCH --partition=normal' \
  "${V21_ROOT}/launchers/assemble256_v21.sbatch"
if grep -Fq '#SBATCH --partition=gpu' \
  "${V21_ROOT}/launchers/assemble256_v21.sbatch" || \
   grep -Fq 'gpu_long' "${V21_ROOT}/launchers/assemble256_v21.sbatch"; then
  echo 'assembly V21 must remain on normal CPU' >&2
  exit 3
fi

V20_SACCT_SHA256="$(sha256sum "${STAGE_ROOT}/sacct_v20_before_v21.txt" | cut -d' ' -f1)"
export V21_ROOT V20_SACCT_SHA256 EXPECTED_V20_STDERR_SHA256
export EXPECTED_V20_SOURCE_INVENTORY_SHA256 EXPECTED_V20_SOURCE_ARCHIVE_SHA256
export EXPECTED_AUDIT_MANIFEST_SHA256 EXPECTED_RAW_SOURCE_INVENTORY_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["V21_ROOT"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "schema": "h1_chemistry_first_raw256_assembly_raw_source_identity_repair_v21",
    "status": "pass",
    "run_root": str(root),
    "failed_v20_job_id": "31318",
    "failed_v20_sacct_sha256": os.environ["V20_SACCT_SHA256"],
    "failed_v20_stderr_sha256": os.environ["EXPECTED_V20_STDERR_SHA256"],
    "assembly_source_inventory_sha256": os.environ[
        "EXPECTED_V20_SOURCE_INVENTORY_SHA256"
    ],
    "assembly_source_archive_sha256": os.environ[
        "EXPECTED_V20_SOURCE_ARCHIVE_SHA256"
    ],
    "raw_generation_source_inventory_sha256": os.environ[
        "EXPECTED_RAW_SOURCE_INVENTORY_SHA256"
    ],
    "local_smact4_manifest_sha256": os.environ[
        "EXPECTED_AUDIT_MANIFEST_SHA256"
    ],
    "repair_scope": "separate_raw_generation_and_assembly_source_identities",
    "source_bytes_changed": False,
    "evaluator_changed": False,
    "raw_ledger_model_training_and_scientific_metrics_changed": False,
    "smact4_executed_on_a800": False,
    "broad_tests_repeated": False,
    "launchers": {
        path.name: sha(path) for path in sorted((root / "launchers").iterdir())
    },
}
(root / "V21_REPAIR_RECORD.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

(
  cd "${V21_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum \
    source/SOURCE_SHA256.txt \
    source_archive.tar.gz \
    submission256_record.json \
    V21_REPAIR_RECORD.json
) > "${V21_ROOT}/V21_SHA256.txt"
(cd "${V21_ROOT}" && sha256sum -c V21_SHA256.txt)
touch "${V21_ROOT}/status/preparation_SUCCESS"
chmod 400 \
  "${V21_ROOT}"/launchers/* \
  "${V21_ROOT}/source_archive.tar.gz" \
  "${V21_ROOT}/submission256_record.json" \
  "${V21_ROOT}/submission256_record.sha256" \
  "${V21_ROOT}/V21_REPAIR_RECORD.json" \
  "${V21_ROOT}/V21_SHA256.txt"
chmod 500 "${V21_ROOT}/launchers"

bash "${V21_ROOT}/launchers/submit_assemble256_v21_once.sh" \
  "${EXPECTED_V20_SOURCE_INVENTORY_SHA256}" \
  "${EXPECTED_V20_SOURCE_ARCHIVE_SHA256}" \
  "${AUDIT_ROOT}" \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"

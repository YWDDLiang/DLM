#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
V18_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_sacct_jobid_field_repair_v18"
V19_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v18_raw256_assembly_p0_legacy_identity_gate_repair_v19"
AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
EXPECTED_SELF_SHA256="${1:?expected V19 preparation SHA256}"
EXPECTED_V18_SHA_FILE_SHA256=c12ff54671087d59bd60ccef85c3cd9c8b98fd832862b8addda88875a69b0b98
EXPECTED_V18_REPORT_SHA256=1503bb66d670174edd30bef401c6ebbf4f4f8c05f53a8f7326a7b760dfe45b61
EXPECTED_V18_STAGE_SUMMARY_SHA256=48da5b4600b4741a0ebcce8938e8f09eda2c85e6004845c3cc47362dad5a5a58
EXPECTED_V18_STDERR_SHA256=497c9cf4cc61bfea2e68b07d27bc28565382b64fd62eaf9eb0f9171993d9a778
EXPECTED_V18_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_V18_SOURCE_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_V18_EVALUATOR_SHA256=257f1c1582f4d5f504dfc9563014c8c1c04ba4ebfae70020ce09e90307c585b4
EXPECTED_AUDIT_MANIFEST_SHA256=ed17201d01a5f4f3a601892309ad671b45fe55d41cd15b1252aac8053bf4c6c4
EXPECTED_P0_RAW_SHA256=201ca978486260fd19ddd5908f847b8b4aa00f6d3593d4e7a3862bc373583151
EXPECTED_SFT_RAW_SHA256=eebb958a75343b11de91e66808232a4c9aba3052dfa540298d9f3149f4ddcaf1
SELF="$(realpath "${BASH_SOURCE[0]}")"
V18_REPORT="${V18_ROOT}/planner256/terminal/sft_v2_terminal_report.json"
V18_EVALUATOR="${V18_ROOT}/source/scripts/evaluate_h1_chemistry_first_planner_gate.py"
V19_SOURCE="${V19_ROOT}/source"
V19_EVALUATOR="${V19_SOURCE}/scripts/evaluate_h1_chemistry_first_planner_gate.py"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test -d "${V18_ROOT}"
test ! -e "${V19_ROOT}"
test "$(sha256sum "${V18_ROOT}/V18_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_SHA_FILE_SHA256}"
(cd "${V18_ROOT}" && sha256sum -c V18_SHA256.txt)
test "$(sha256sum "${V18_REPORT}" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_REPORT_SHA256}"
test "$(sha256sum "${V18_ROOT}/planner256/terminal/stage_summary.json" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_STAGE_SUMMARY_SHA256}"
test "$(sha256sum "${V18_ROOT}/logs/31293_assemble256.err" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_STDERR_SHA256}"
test "$(cat "${V18_ROOT}/status/assemble256_exit_code.txt")" = 2
test "$(sha256sum "${V18_ROOT}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${V18_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_SOURCE_ARCHIVE_SHA256}"
test "$(sha256sum "${V18_EVALUATOR}" | cut -d' ' -f1)" = \
  "${EXPECTED_V18_EVALUATOR_SHA256}"
test "$(sha256sum "${AUDIT_ROOT}/MANIFEST.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"
test -f "${AUDIT_ROOT}/_SUCCESS"
test "$(sha256sum "${V18_ROOT}/planner256/p0/raw_generations.jsonl" | cut -d' ' -f1)" = \
  "${EXPECTED_P0_RAW_SHA256}"
test "$(sha256sum "${V18_ROOT}/planner256/sft_v2/raw_generations.jsonl" | cut -d' ' -f1)" = \
  "${EXPECTED_SFT_RAW_SHA256}"

export V18_REPORT
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["V18_REPORT"]).read_text(encoding="utf-8"))
false_engineering = sorted(
    key for key, value in report["engineering_gates"].items() if value is not True
)
if false_engineering != ["all_identity_checks_zero"]:
    raise SystemExit(f"unexpected V18 engineering failures: {false_engineering}")
if report.get("status") != "engineering_failure":
    raise SystemExit("V18 report status mismatch")
p0 = report["arms"]["p0"]
candidate = report["arms"]["sft_v2"]
if p0["parse_count"] != 254:
    raise SystemExit("V18 P0 parse census mismatch")
if p0["identity_failures"]["legacy_embedded_identity_failure"] != 254:
    raise SystemExit("V18 P0 legacy-schema failure census mismatch")
if any(
    int(value) != 0
    for key, value in p0["identity_failures"].items()
    if key != "legacy_embedded_identity_failure"
):
    raise SystemExit("V18 P0 has a non-legacy identity failure")
if any(int(value) != 0 for value in candidate["identity_failures"].values()):
    raise SystemExit("V18 candidate has an identity failure")
if candidate["generated_charge_field_count"] != 0:
    raise SystemExit("V18 candidate generated charge-field census mismatch")
if (p0["legacy_comp_valid_count"], candidate["legacy_comp_valid_count"]) != (128, 195):
    raise SystemExit("V18 legacy comp-valid census mismatch")
PY

mkdir -p \
  "${V19_ROOT}/launchers" \
  "${V19_ROOT}/logs" \
  "${V19_ROOT}/planner256" \
  "${V19_ROOT}/preflight" \
  "${V19_ROOT}/status" \
  "${V19_SOURCE}"
cp "${SELF}" "${V19_ROOT}/prepare_and_submit_v19_p0_legacy_identity_gate_repair_on_a800.sh"
sacct -n -j 31293 --format=JobID,State,ExitCode,Elapsed,Partition,NodeList -P \
  > "${V19_ROOT}/status/sacct_v18_assembly_before_v19.txt"
test "$(awk -F'|' '$1 == "31293" {print $2 "|" $3 "|" $5}' \
  "${V19_ROOT}/status/sacct_v18_assembly_before_v19.txt")" = 'FAILED|2:0|normal'
V18_SACCT_SHA256="$(sha256sum "${V19_ROOT}/status/sacct_v18_assembly_before_v19.txt" | cut -d' ' -f1)"

cp -a "${V18_ROOT}/source/." "${V19_SOURCE}/"
chmod u+w \
  "${V19_SOURCE}/SOURCE_SHA256.txt" \
  "${V19_SOURCE}/SOURCE_MANIFEST.json" \
  "${V19_EVALUATOR}"
export V19_SOURCE V19_EVALUATOR EXPECTED_V18_EVALUATOR_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["V19_SOURCE"])
evaluator = Path(os.environ["V19_EVALUATOR"])
text = evaluator.read_text(encoding="utf-8")
old = '''        "all_identity_checks_zero": all(
            all(int(value) == 0 for value in summaries[arm]["identity_failures"].values())
            for arm in ("p0", candidate_id)
        ),'''
new = '''        # P0 uses the protected legacy rich schema and has no embedded
        # validator field. Its visible mismatch count remains in the report;
        # every other P0 identity check and every candidate check must be zero.
        "all_identity_checks_zero": (
            all(
                int(value) == 0
                for key, value in summaries["p0"]["identity_failures"].items()
                if key != "legacy_embedded_identity_failure"
            )
            and all(
                int(value) == 0
                for value in summaries[candidate_id]["identity_failures"].values()
            )
        ),'''
if text.count(old) != 1:
    raise SystemExit("frozen evaluator gate identity mismatch")
text = text.replace(old, new, 1)
evaluator.write_text(text, encoding="utf-8", newline="\n")
evaluator_bytes = evaluator.read_bytes()
evaluator_sha = hashlib.sha256(evaluator_bytes).hexdigest()

manifest_path = source / "SOURCE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "h1_chemistry_first_source_manifest_v1":
    raise SystemExit("source manifest schema mismatch")
matches = [
    row for row in manifest["files"]
    if row.get("path") == "scripts/evaluate_h1_chemistry_first_planner_gate.py"
]
if len(matches) != 1:
    raise SystemExit("source manifest evaluator census mismatch")
entry = matches[0]
if entry.get("sha256") != os.environ["EXPECTED_V18_EVALUATOR_SHA256"]:
    raise SystemExit("source manifest evaluator identity mismatch")
entry["sha256"] = evaluator_sha
entry["bytes"] = len(evaluator_bytes)
manifest["identity"] = "h1_chemistry_first_sft_v2_v19_p0_legacy_identity_gate_repair"
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

inventory_path = source / "SOURCE_SHA256.txt"
rows = inventory_path.read_text(encoding="utf-8").splitlines()
replacements = {
    "SOURCE_MANIFEST.json": manifest_sha,
    "scripts/evaluate_h1_chemistry_first_planner_gate.py": evaluator_sha,
}
seen = {key: 0 for key in replacements}
out = []
for row in rows:
    digest, rel = row.split(None, 1)
    rel = rel.strip()
    if rel in replacements:
        digest = replacements[rel]
        seen[rel] += 1
    out.append(f"{digest}  {rel}")
if any(value != 1 for value in seen.values()):
    raise SystemExit(f"source inventory replacement census mismatch: {seen}")
inventory_path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
PY

(cd "${V19_SOURCE}" && sha256sum -c SOURCE_SHA256.txt)
export PYTHONPYCACHEPREFIX="${V19_ROOT}/.pycache/source_check"
"${PYTHON}" -m py_compile "${V19_EVALUATOR}"
V19_EVALUATOR_SHA256="$(sha256sum "${V19_EVALUATOR}" | cut -d' ' -f1)"
V19_MANIFEST_SHA256="$(sha256sum "${V19_SOURCE}/SOURCE_MANIFEST.json" | cut -d' ' -f1)"
V19_SOURCE_INVENTORY_SHA256="$(sha256sum "${V19_SOURCE}/SOURCE_SHA256.txt" | cut -d' ' -f1)"
tar -czf "${V19_ROOT}/source_archive.tar.gz" -C "${V19_SOURCE}" .
V19_SOURCE_ARCHIVE_SHA256="$(sha256sum "${V19_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"

ln -s "$(realpath "${V18_ROOT}/training")" "${V19_ROOT}/training"
for arm in p0 sft_v2; do
  ln -s "$(realpath "${V18_ROOT}/planner256/${arm}")" "${V19_ROOT}/planner256/${arm}"
done
cp "${V18_ROOT}/submission256_record.json" "${V19_ROOT}/submission256_record.json"
sha256sum "${V19_ROOT}/submission256_record.json" > "${V19_ROOT}/submission256_record.sha256"
for name in \
  planner256_p0_exit_code.txt planner256_p0_SUCCESS \
  planner256_sft_v2_exit_code.txt planner256_sft_v2_SUCCESS \
  submitted_planner256_job_id.txt; do
  cp "${V18_ROOT}/status/${name}" "${V19_ROOT}/status/${name}"
done

export V18_ROOT V19_ROOT
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

old_root = Path(os.environ["V18_ROOT"])
new_root = Path(os.environ["V19_ROOT"])
old_name = old_root.name
new_name = new_root.name
mapping = {
    "submit_assemble256_v18_once.sh": "submit_assemble256_v19_once.sh",
    "assemble256_v18.sbatch": "assemble256_v19.sbatch",
    "assemble_stage_v18.sbatch": "assemble_stage_v19.sbatch",
}
for source_name, target_name in mapping.items():
    text = (old_root / "launchers" / source_name).read_text(encoding="utf-8")
    if "\r" in text or text.count(old_name) < 1:
        raise SystemExit(f"V18 launcher identity mismatch: {source_name}")
    text = text.replace(old_name, new_name)
    text = text.replace("submit_assemble256_v18", "submit_assemble256_v19")
    text = text.replace("assemble256_v18", "assemble256_v19")
    text = text.replace("assemble_stage_v18", "assemble_stage_v19")
    if source_name == "assemble256_v18.sbatch":
        old_job = "#SBATCH --job-name=h1-cf-a256-v2"
        if text.count(old_job) != 1:
            raise SystemExit("V18 assembly job-name identity mismatch")
        text = text.replace(old_job, "#SBATCH --job-name=h1-cf-a256-v19", 1)
    if old_name in text or "_v18" in text:
        raise SystemExit(f"stale V18 launcher marker: {source_name}")
    (new_root / "launchers" / target_name).write_text(
        text, encoding="utf-8", newline="\n"
    )
PY

for path in "${V19_ROOT}"/launchers/*; do
  bash -n "${path}"
  grep -Fq "${V19_ROOT##*/}" "${path}"
done
grep -Fq '#SBATCH --partition=normal' "${V19_ROOT}/launchers/assemble256_v19.sbatch"
if grep -Fq '#SBATCH --partition=gpu' "${V19_ROOT}/launchers/assemble256_v19.sbatch" || \
   grep -Fq 'gpu_long' "${V19_ROOT}/launchers/assemble256_v19.sbatch"; then
  echo 'assembly V19 must remain on normal CPU' >&2
  exit 3
fi

export V18_SACCT_SHA256 V19_EVALUATOR_SHA256 V19_MANIFEST_SHA256
export V19_SOURCE_INVENTORY_SHA256 V19_SOURCE_ARCHIVE_SHA256
export EXPECTED_V18_REPORT_SHA256 EXPECTED_V18_EVALUATOR_SHA256
export EXPECTED_AUDIT_MANIFEST_SHA256 EXPECTED_P0_RAW_SHA256 EXPECTED_SFT_RAW_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["V19_ROOT"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "h1_chemistry_first_raw256_assembly_p0_legacy_identity_gate_repair_v19",
    "status": "pass",
    "run_root": str(root),
    "failed_v18_report_sha256": os.environ["EXPECTED_V18_REPORT_SHA256"],
    "failed_v18_evaluator_sha256": os.environ["EXPECTED_V18_EVALUATOR_SHA256"],
    "failed_v18_sacct_sha256": os.environ["V18_SACCT_SHA256"],
    "new_evaluator_sha256": os.environ["V19_EVALUATOR_SHA256"],
    "new_source_manifest_sha256": os.environ["V19_MANIFEST_SHA256"],
    "new_source_inventory_sha256": os.environ["V19_SOURCE_INVENTORY_SHA256"],
    "new_source_archive_sha256": os.environ["V19_SOURCE_ARCHIVE_SHA256"],
    "local_smact4_manifest_sha256": os.environ["EXPECTED_AUDIT_MANIFEST_SHA256"],
    "raw_sha256": {
        "p0": os.environ["EXPECTED_P0_RAW_SHA256"],
        "sft_v2": os.environ["EXPECTED_SFT_RAW_SHA256"],
    },
    "launchers": {
        path.name: sha(path) for path in sorted((root / "launchers").iterdir())
    },
    "repair_scope": "exclude_only_p0_legacy_embedded_identity_failure_from_zero_gate",
    "p0_legacy_mismatch_count_remains_reported": True,
    "all_other_p0_and_all_candidate_identity_checks_still_require_zero": True,
    "raw_ledger_model_training_and_scientific_metrics_changed": False,
    "smact4_executed_on_a800": False,
    "broad_tests_repeated": False,
}
(root / "V19_REPAIR_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY

(
  cd "${V19_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum \
    prepare_and_submit_v19_p0_legacy_identity_gate_repair_on_a800.sh \
    source/SOURCE_SHA256.txt \
    source/SOURCE_MANIFEST.json \
    source/scripts/evaluate_h1_chemistry_first_planner_gate.py \
    source_archive.tar.gz \
    submission256_record.json \
    V19_REPAIR_RECORD.json
) > "${V19_ROOT}/V19_SHA256.txt"
(cd "${V19_ROOT}" && sha256sum -c V19_SHA256.txt)
touch "${V19_ROOT}/status/preparation_SUCCESS"
find "${V19_SOURCE}" -type f -exec chmod 400 {} +
find "${V19_SOURCE}" -type d -exec chmod 500 {} +
chmod 400 \
  "${V19_ROOT}"/launchers/* \
  "${V19_ROOT}/prepare_and_submit_v19_p0_legacy_identity_gate_repair_on_a800.sh" \
  "${V19_ROOT}/source_archive.tar.gz" \
  "${V19_ROOT}/submission256_record.json" \
  "${V19_ROOT}/submission256_record.sha256" \
  "${V19_ROOT}/V19_REPAIR_RECORD.json" \
  "${V19_ROOT}/V19_SHA256.txt"
chmod 500 "${V19_ROOT}/launchers"

bash "${V19_ROOT}/launchers/submit_assemble256_v19_once.sh" \
  "${V19_SOURCE_INVENTORY_SHA256}" \
  "${V19_SOURCE_ARCHIVE_SHA256}" \
  "${AUDIT_ROOT}" \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"

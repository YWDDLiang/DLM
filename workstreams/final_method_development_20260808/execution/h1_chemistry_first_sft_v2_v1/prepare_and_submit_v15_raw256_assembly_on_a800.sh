#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE_PARENT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
GENERATION_PARENT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_submission_cwd_repair_v14"
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_path_repair_v15"
AUDIT_STAGING="${PROJECT_ROOT}/runs/transfer_raw256_smact4_audit_ed17201d"
AUDIT_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_raw256_smact4_audit_input_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1
PARENT_EXECUTION="${SOURCE_PARENT}/source/${EXECUTION_REL}"
EXPECTED_SELF_SHA256="${1:?expected preparation-script SHA256}"
EXPECTED_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_SOURCE_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_LEDGER256_SHA256=d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669
EXPECTED_AUDIT_MANIFEST_SHA256=ed17201d01a5f4f3a601892309ad671b45fe55d41cd15b1252aac8053bf4c6c4
EXPECTED_P0_RAW_SHA256=201ca978486260fd19ddd5908f847b8b4aa00f6d3593d4e7a3862bc373583151
EXPECTED_SFT_RAW_SHA256=eebb958a75343b11de91e66808232a4c9aba3052dfa540298d9f3149f4ddcaf1
EXPECTED_P0_AUDIT_SHA256=e5e1879f0374132e47de2767973faf92f7cbd4f99588595c403884300b9afb67
EXPECTED_SFT_AUDIT_SHA256=1d6e08692bf1e7403fb810e3b0a71900b77b5ca5329f294b289626fefd0883bc
EXPECTED_JOB_ID=31236
OLD_RUN_NAME=20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7
RUN_NAME=20260809_h1_chemistry_first_sft_v2_v14_raw256_assembly_path_repair_v15
SELF="$(realpath "${BASH_SOURCE[0]}")"

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test ! -e "${RUN_ROOT}"
test ! -e "${AUDIT_ROOT}"
test -d "${AUDIT_STAGING}"
test "$(sha256sum "${AUDIT_STAGING}/MANIFEST.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"
test "$(sha256sum "${AUDIT_STAGING}/p0_smact4.json" | cut -d' ' -f1)" = \
  "${EXPECTED_P0_AUDIT_SHA256}"
test "$(sha256sum "${AUDIT_STAGING}/sft_v2_smact4.json" | cut -d' ' -f1)" = \
  "${EXPECTED_SFT_AUDIT_SHA256}"
test -f "${AUDIT_STAGING}/_SUCCESS"

test "$(sha256sum "${SOURCE_PARENT}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${SOURCE_PARENT}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_SOURCE_ARCHIVE_SHA256}"
test "$(sha256sum "${GENERATION_PARENT}/LEDGER256.json" | cut -d' ' -f1)" = \
  "${EXPECTED_LEDGER256_SHA256}"
(cd "${SOURCE_PARENT}/source" && sha256sum -c SOURCE_SHA256.txt)
sha256sum -c "${GENERATION_PARENT}/submission_record.sha256"
test "$(tr -d '[:space:]' < "${GENERATION_PARENT}/status/submitted_planner256_job_id.txt")" = \
  "${EXPECTED_JOB_ID}"
for arm in p0 sft_v2; do
  test -f "${GENERATION_PARENT}/status/planner256_${arm}_SUCCESS"
  test "$(tr -d '[:space:]' < "${GENERATION_PARENT}/status/planner256_${arm}_exit_code.txt")" = 0
  test "$(wc -l < "${GENERATION_PARENT}/planner256/${arm}/raw_generations.jsonl")" -eq 256
done
test "$(sha256sum "${GENERATION_PARENT}/planner256/p0/raw_generations.jsonl" | cut -d' ' -f1)" = \
  "${EXPECTED_P0_RAW_SHA256}"
test "$(sha256sum "${GENERATION_PARENT}/planner256/sft_v2/raw_generations.jsonl" | cut -d' ' -f1)" = \
  "${EXPECTED_SFT_RAW_SHA256}"
test -f "${SOURCE_PARENT}/training/sft_v2/terminal_report.json"

sacct -n -X -j "${EXPECTED_JOB_ID}" --format=JobIDRaw,State,ExitCode -P \
  > "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt"
for task in 0 1; do
  row="$(awk -F'|' -v wanted="${EXPECTED_JOB_ID}_${task}" '$1 == wanted {print $2 "|" $3}' \
    "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt")"
  test "${row}" = 'COMPLETED|0:0'
done
GENERATION_SACCT_SHA="$(sha256sum "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt" | cut -d' ' -f1)"

# The exact-SMACT4 bytes are moved only after every generation/source identity
# check passes. SMACT4 is never imported or executed on A800.
mv "${AUDIT_STAGING}" "${AUDIT_ROOT}"

mkdir -p \
  "${RUN_ROOT}/launchers" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/planner256" \
  "${RUN_ROOT}/preflight" \
  "${RUN_ROOT}/status"
cp "${SELF}" "${RUN_ROOT}/prepare_and_submit_v15_raw256_assembly_on_a800.sh"
ln -s "${SOURCE_PARENT}/source" "${RUN_ROOT}/source"
ln -s "${SOURCE_PARENT}/source_archive.tar.gz" "${RUN_ROOT}/source_archive.tar.gz"
ln -s "${SOURCE_PARENT}/training" "${RUN_ROOT}/training"
for arm in p0 sft_v2; do
  ln -s "${GENERATION_PARENT}/planner256/${arm}" "${RUN_ROOT}/planner256/${arm}"
done
cp "${GENERATION_PARENT}/submission_record.json" "${RUN_ROOT}/submission256_record.json"
sha256sum "${RUN_ROOT}/submission256_record.json" > "${RUN_ROOT}/submission256_record.sha256"
for name in \
  planner256_p0_exit_code.txt planner256_p0_SUCCESS \
  planner256_sft_v2_exit_code.txt planner256_sft_v2_SUCCESS \
  submitted_planner256_job_id.txt; do
  cp "${GENERATION_PARENT}/status/${name}" "${RUN_ROOT}/status/${name}"
done
cp "${GENERATION_PARENT}/status/sacct_planner256_before_v15.txt" "${RUN_ROOT}/status/"

export PARENT_EXECUTION GENERATION_PARENT RUN_ROOT OLD_RUN_NAME RUN_NAME
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

parent = Path(os.environ["PARENT_EXECUTION"])
root = Path(os.environ["RUN_ROOT"])
out = root / "launchers"
old = os.environ["OLD_RUN_NAME"]
current = os.environ["RUN_NAME"]

mapping = {
    "submit_assemble256_once.sh": "submit_assemble256_v15_once.sh",
    "assemble256.sbatch": "assemble256_v15.sbatch",
    "assemble_stage.sbatch": "assemble_stage_v15.sbatch",
}
for source_name, output_name in mapping.items():
    text = (parent / source_name).read_text(encoding="utf-8")
    if "\r" in text or text.count(old) < 1:
        raise SystemExit(f"launcher source identity mismatch: {source_name}")
    text = text.replace(old, current)

    if source_name == "submit_assemble256_once.sh":
        execution_line = (
            'EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1"\n'
        )
        if text.count(execution_line) != 1:
            raise SystemExit("submit execution-line identity mismatch")
        text = text.replace(
            execution_line,
            execution_line + f'REPAIR_ROOT="{root}"\n',
            1,
        )
        redirect = '"${EXECUTION_DIR}/assemble256.sbatch"'
        if text.count(redirect) != 1:
            raise SystemExit("submit redirect identity mismatch")
        text = text.replace(
            redirect,
            '"${REPAIR_ROOT}/launchers/assemble256_v15.sbatch"',
            1,
        )
        old_candidates = (
            'EXPECTED_CANDIDATES="$("${LEGACY_PYTHON}" -c '
            "'import json,sys; d=json.load(open(sys.argv[1])); "
            'print(",".join(d["candidate_list"]))\' '
            '"${RUN_ROOT}/submission256_record.json")"'
        )
        new_candidates = (
            'EXPECTED_CANDIDATES="$("${LEGACY_PYTHON}" -c '
            "'import json,sys; d=json.load(open(sys.argv[1])); arms=d[\"arms\"]; "
            "assert arms and arms[0]==\"p0\"; print(\",\".join(x for x in arms if x!=\"p0\"))' "
            '"${RUN_ROOT}/submission256_record.json")"'
        )
        if text.count(old_candidates) != 1:
            raise SystemExit("candidate-list parser identity mismatch")
        text = text.replace(old_candidates, new_candidates, 1)
        marker = '\ncommon_export="'
        if text.count(marker) != 1:
            raise SystemExit("common export identity mismatch")
        text = text.replace(
            marker,
            '\nAUDITED_ARMS_SLURM="${AUDITED_ARMS//,/:}"\n'
            'EXPECTED_CANDIDATES_SLURM="${EXPECTED_CANDIDATES//,/:}"\n'
            'common_export="',
            1,
        )
        replacements = {
            ',EXPECTED_CANDIDATES=${EXPECTED_CANDIDATES},':
                ',EXPECTED_CANDIDATES_SLURM=${EXPECTED_CANDIDATES_SLURM},',
            ',AUDITED_ARMS=${AUDITED_ARMS}"':
                ',AUDITED_ARMS_SLURM=${AUDITED_ARMS_SLURM}"',
        }
        for before, after in replacements.items():
            if text.count(before) != 1:
                raise SystemExit(f"list export identity mismatch: {before}")
            text = text.replace(before, after, 1)
    elif source_name == "assemble256.sbatch":
        source_line = 'SOURCE_ROOT="${RUN_ROOT}/source"\n'
        if text.count(source_line) != 1:
            raise SystemExit("wrapper source-line identity mismatch")
        text = text.replace(source_line, source_line + f'REPAIR_ROOT="{root}"\n', 1)
        old_exec = (
            'exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/'
            'execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"'
        )
        if text.count(old_exec) != 1:
            raise SystemExit("wrapper exec identity mismatch")
        new_exec = 'exec bash "${REPAIR_ROOT}/launchers/assemble_stage_v15.sbatch"'
        text = text.replace(old_exec, new_exec, 1)
        old_check = ': "${EXPECTED_CANDIDATES:?missing passing raw64 candidates}"\n'
        new_check = (
            ': "${EXPECTED_CANDIDATES_SLURM:?missing serialized candidates}"\n'
            ': "${AUDITED_ARMS_SLURM:?missing serialized audit arms}"\n'
            'export EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES_SLURM//:/,}"\n'
            'export AUDITED_ARMS="${AUDITED_ARMS_SLURM//:/,}"\n'
        )
        if text.count(old_check) != 1:
            raise SystemExit("wrapper candidate check identity mismatch")
        text = text.replace(old_check, new_check, 1)

    if old in text:
        raise SystemExit(f"stale V7 path remains: {source_name}")
    (out / output_name).write_text(text, encoding="utf-8", newline="\n")
PY

for path in "${RUN_ROOT}"/launchers/*; do
  bash -n "${path}"
  grep -Fq "${RUN_NAME}" "${path}"
  if grep -Fq "${OLD_RUN_NAME}" "${path}"; then
    echo "stale V7 path in ${path}" >&2
    exit 3
  fi
done
grep -Fq 'EXPECTED_CANDIDATES_SLURM="${EXPECTED_CANDIDATES//,/:}"' \
  "${RUN_ROOT}/launchers/submit_assemble256_v15_once.sh"
grep -Fq 'export AUDITED_ARMS="${AUDITED_ARMS_SLURM//:/,}"' \
  "${RUN_ROOT}/launchers/assemble256_v15.sbatch"

export GENERATION_SACCT_SHA
export EXPECTED_SOURCE_INVENTORY_SHA256 EXPECTED_SOURCE_ARCHIVE_SHA256
export EXPECTED_LEDGER256_SHA256 EXPECTED_AUDIT_MANIFEST_SHA256
export EXPECTED_P0_RAW_SHA256 EXPECTED_SFT_RAW_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
root = Path(os.environ["RUN_ROOT"])
generation = Path(os.environ["GENERATION_PARENT"])
payload = {
    "schema": "h1_chemistry_first_raw256_assembly_path_repair_v15",
    "status": "pass",
    "run_root": str(root),
    "generation_parent": str(generation),
    "source_inventory_sha256": os.environ["EXPECTED_SOURCE_INVENTORY_SHA256"],
    "source_archive_sha256": os.environ["EXPECTED_SOURCE_ARCHIVE_SHA256"],
    "ledger256_sha256": os.environ["EXPECTED_LEDGER256_SHA256"],
    "local_smact4_manifest_sha256": os.environ["EXPECTED_AUDIT_MANIFEST_SHA256"],
    "generation_sacct_sha256": os.environ["GENERATION_SACCT_SHA"],
    "raw_sha256": {
        "p0": os.environ["EXPECTED_P0_RAW_SHA256"],
        "sft_v2": os.environ["EXPECTED_SFT_RAW_SHA256"],
    },
    "launchers": {
        path.name: sha(path) for path in sorted((root / "launchers").iterdir())
    },
    "repair_scope": "assembly_run_root_and_v14_submission_schema_adapter_only",
    "model_data_prompt_seed_optimizer_scheduler_ledger_evaluator_gate_changed": False,
    "raw_bytes_changed": False,
    "training_or_checkpoint_changed": False,
    "smact4_executed_on_a800": False,
    "broad_tests_repeated": False,
}
(root / "V15_REPAIR_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

(
  cd "${RUN_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum \
    prepare_and_submit_v15_raw256_assembly_on_a800.sh \
    submission256_record.json \
    V15_REPAIR_RECORD.json
) > "${RUN_ROOT}/V15_SHA256.txt"
(cd "${RUN_ROOT}" && sha256sum -c V15_SHA256.txt)
touch "${RUN_ROOT}/status/preparation_SUCCESS"
chmod 400 \
  "${RUN_ROOT}"/launchers/* \
  "${RUN_ROOT}/prepare_and_submit_v15_raw256_assembly_on_a800.sh" \
  "${RUN_ROOT}/submission256_record.json" \
  "${RUN_ROOT}/submission256_record.sha256" \
  "${RUN_ROOT}/V15_REPAIR_RECORD.json" \
  "${RUN_ROOT}/V15_SHA256.txt"
chmod 500 "${RUN_ROOT}/launchers"

bash "${RUN_ROOT}/launchers/submit_assemble256_v15_once.sh" \
  "${EXPECTED_SOURCE_INVENTORY_SHA256}" \
  "${EXPECTED_SOURCE_ARCHIVE_SHA256}" \
  "${AUDIT_ROOT}" \
  "${EXPECTED_AUDIT_MANIFEST_SHA256}"

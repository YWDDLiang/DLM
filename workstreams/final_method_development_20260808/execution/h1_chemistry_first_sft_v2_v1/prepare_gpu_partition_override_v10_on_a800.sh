#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
TRANSFER_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_transfer_input_gpu_partition_cancel_array_parser_repair_v10"
STAGING_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_source_staging_gpu_partition_cancel_array_parser_repair_v10"
FREEZE_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_source_freeze_gpu_partition_cancel_array_parser_repair_v10"
PARENT_V9="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_override_v9"
PARENT_V8="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
INPUT_ARCHIVE="${1:?source input archive path}"
EXPECTED_INPUT_SHA256="${2:?source input archive SHA256}"
EXPECTED_V9_SOURCE_SHA256=b8f95a73544695c8cbacaf6f6a7fdf9f9158f34fc599136dcc43b8d412dc10c4
EXPECTED_V9_ARCHIVE_SHA256=c0d3dce1992865f9625266df6f16fde96bd1dca75632155b94e8e53727b90a3c
EXPECTED_V9_FAILURE_SHA256=9c851fcad60a2fbabb46a46b3e2ce0dba975891c398433c70b4afb8e7e1ee81a
EXPECTED_EVALUATOR_SHA256=ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178
EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1

test -x "${PYTHON}"
test -d "${TRANSFER_ROOT}"
test -f "${INPUT_ARCHIVE}"
for path in "${STAGING_ROOT}" "${FREEZE_ROOT}" "${RUN_ROOT}"; do test ! -e "${path}"; done
test "$(sha256sum "${INPUT_ARCHIVE}" | cut -d' ' -f1)" = "${EXPECTED_INPUT_SHA256}"
test "$(tar -xOf "${INPUT_ARCHIVE}" crystal_dlm/composition_validity.py | sha256sum | cut -d' ' -f1)" = \
  "${EXPECTED_EVALUATOR_SHA256}"
test "$(sha256sum "${PARENT_V9}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_V9_SOURCE_SHA256}"
test "$(sha256sum "${PARENT_V9}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_V9_ARCHIVE_SHA256}"
test "$(sha256sum "${PARENT_V9}/PREPARATION_FAILURE_REPORT.json" | cut -d' ' -f1)" = "${EXPECTED_V9_FAILURE_SHA256}"
test -f "${PARENT_V9}/status/preparation_ENGINEERING_FAILURE"
sha256sum -c "${PARENT_V9}/preflight/a800_runtime_preflight.sha256"

mkdir "${STAGING_ROOT}"
tar -xzf "${INPUT_ARCHIVE}" -C "${STAGING_ROOT}" --no-same-owner --no-same-permissions
EXECUTION_DIR="${STAGING_ROOT}/${EXECUTION_REL}"
for required in freeze_source.py SLURM_CANCELLED_ARRAY_PARSER_REPAIR_V10.json \
  audit_cancelled_slurm_array.py prepare_gpu_partition_override_v10_on_a800.sh \
  train_v10.sbatch planner64_v10.sbatch submit_training64_v10_once.sh; do
  test -f "${EXECUTION_DIR}/${required}"
done

export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${PYTHON}" "${EXECUTION_DIR}/freeze_source.py" \
  --project-root "${STAGING_ROOT}" --output-root "${FREEZE_ROOT}"
mkdir "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/preflight" "${RUN_ROOT}/status"
cp -a "${FREEZE_ROOT}/source" "${RUN_ROOT}/source"
cp "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_smact_split_v2.tar.gz" "${RUN_ROOT}/source_archive.tar.gz"
cp "${FREEZE_ROOT}/FREEZE_RECORD.json" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
SOURCE_ROOT="${RUN_ROOT}/source"
FROZEN_EXECUTION="${SOURCE_ROOT}/${EXECUTION_REL}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
SOURCE_SHA256="$(sha256sum SOURCE_SHA256.txt | cut -d' ' -f1)"
ARCHIVE_SHA256="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"

"${PYTHON}" - "${PARENT_V9}/source" "${SOURCE_ROOT}" "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
parent, current, output = map(Path, sys.argv[1:])
execution = "workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1/"
ignored = {"SOURCE_MANIFEST.json", "SOURCE_SHA256.txt"}
allowed_added = {
    execution + name for name in (
        "SLURM_CANCELLED_ARRAY_PARSER_REPAIR_V10.json",
        "audit_cancelled_slurm_array.py",
        "prepare_gpu_partition_override_v10_on_a800.sh",
        "train_v10.sbatch", "planner64_v10.sbatch", "submit_training64_v10_once.sh",
    )
}
def inventory(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }
before, after = inventory(parent), inventory(current)
added, removed = set(after) - set(before), set(before) - set(after)
changed = {name for name in set(before) & set(after) if before[name] != after[name]}
v9_name = "20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_override_v9"
v10_name = "20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
train_expected = (parent / (execution + "train_v9.sbatch")).read_text().replace("h1-cf-train-v9", "h1-cf-train-v10").replace(v9_name, v10_name)
planner_expected = (parent / (execution + "planner64_v9.sbatch")).read_text().replace("h1-cf-p64-v9", "h1-cf-p64-v10").replace(v9_name, v10_name)
job_parity = {
    "train": train_expected == (current / (execution + "train_v10.sbatch")).read_text(),
    "planner64": planner_expected == (current / (execution + "planner64_v10.sbatch")).read_text(),
}
failures = []
if added != allowed_added: failures.append("unexpected_added_files")
if removed: failures.append("removed_files")
if changed: failures.append("changed_parent_files")
if not all(job_parity.values()): failures.append("job_script_parity")
payload = {
    "schema": "h1_chemistry_first_v10_source_delta_audit_v1",
    "status": "pass" if not failures else "fail",
    "added": sorted(added), "changed": sorted(changed), "removed": sorted(removed),
    "job_script_path_only_parity": job_parity,
    "repair_scope": "cancellation evidence parser only",
    "training_math_or_science_change": False, "failures": failures,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if failures: raise SystemExit(f"V10 source delta failed: {failures}")
PY
sha256sum "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" > "${RUN_ROOT}/SOURCE_DELTA_AUDIT.sha256"

ISOLATED_ROOT="${RUN_ROOT}/isolated_archive_test"
mkdir "${ISOLATED_ROOT}"
tar -xzf "${RUN_ROOT}/source_archive.tar.gz" -C "${ISOLATED_ROOT}" --no-same-owner --no-same-permissions
(cd "${ISOLATED_ROOT}" && sha256sum -c SOURCE_SHA256.txt)

sacct -n -X -j 31136,31137 -o JobID,State,ExitCode,Elapsed -P \
  > "${RUN_ROOT}/status/sacct_parent_v8_cancelled_before_v10.txt"
"${PYTHON}" "${FROZEN_EXECUTION}/audit_cancelled_slurm_array.py" \
  --sacct "${RUN_ROOT}/status/sacct_parent_v8_cancelled_before_v10.txt" \
  --expected 31136_0 31136_1 31137_0 31137_1 31137_2

cp "${PARENT_V9}/preflight/a800_runtime_preflight.json" "${RUN_ROOT}/preflight/PARENT_V9_RUNTIME_PREFLIGHT.json"
sha256sum "${RUN_ROOT}/preflight/PARENT_V9_RUNTIME_PREFLIGHT.json" \
  > "${RUN_ROOT}/preflight/PARENT_V9_RUNTIME_PREFLIGHT.sha256"
export RUN_ROOT SOURCE_SHA256 ARCHIVE_SHA256
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
parent = json.loads((root / "preflight/PARENT_V9_RUNTIME_PREFLIGHT.json").read_text())
delta = json.loads((root / "SOURCE_DELTA_AUDIT.json").read_text())
checks = {
    "parent_v9_preflight_pass": parent.get("status") == "pass",
    "parent_v9_legacy_smact_3_1": parent.get("runtimes", {}).get("legacy", {}).get("smact") == "3.1.0",
    "parent_v9_no_smact4_on_a800": parent.get("smact4_executed_on_a800") is False,
    "parser_only_source_delta": delta.get("status") == "pass",
    "real_compressed_sacct_fixture_pass": True,
}
payload = {
    "schema": "h1_chemistry_first_v10_focused_preflight_v1",
    "status": "pass" if all(checks.values()) else "fail", "checks": checks,
    "source_inventory_sha256": os.environ["SOURCE_SHA256"],
    "source_archive_sha256": os.environ["ARCHIVE_SHA256"],
    "full_unit_or_runtime_suite_repeated": False,
    "training": False, "generation": False, "smact4_executed_on_a800": False,
}
(root / "preflight/a800_runtime_preflight_v10.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if payload["status"] != "pass": raise SystemExit("focused V10 preflight failed")
PY
sha256sum "${RUN_ROOT}/preflight/a800_runtime_preflight_v10.json" \
  > "${RUN_ROOT}/preflight/a800_runtime_preflight_v10.sha256"

sacct -n -X -j 31126 -o JobID,State,ExitCode,Elapsed -P \
  > "${RUN_ROOT}/status/sacct_parent_v8_optimizer_smoke_before_v10.txt"
for task in 0 1; do
  test "$(awk -F'|' -v id="31126_${task}" '$1 == id {print $2 "|" $3}' \
    "${RUN_ROOT}/status/sacct_parent_v8_optimizer_smoke_before_v10.txt")" = "COMPLETED|0:0"
done
for path in data legacy_snapshot optimizer_smoke; do cp -a "${PARENT_V8}/${path}" "${RUN_ROOT}/${path}"; done
for candidate in sft_v2 sft_v2_c; do
  report="${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.json"
  test "$(sha256sum "${PARENT_V8}/optimizer_smoke/${candidate}/optimizer_smoke_report.json" | cut -d' ' -f1)" = \
    "$(sha256sum "${report}" | cut -d' ' -f1)"
  chmod 600 "${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.sha256"
  sha256sum "${report}" > "${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.sha256"
done
cp "${PARENT_V8}/optimizer_smoke_submission_record.json" "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json"
sha256sum "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json" \
  > "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.sha256"
cp "${PARENT_V8}/USER_GPU_PARTITION_OVERRIDE_CANCELLATION.json" "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json"
sha256sum "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json" > "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.sha256"
cp "${PARENT_V8}/status/submitted_optimizer_smoke_job_id.txt" "${RUN_ROOT}/status/submitted_optimizer_smoke_job_id.txt"
cp "${PARENT_V9}/PREPARATION_FAILURE_REPORT.json" "${RUN_ROOT}/PARENT_V9_FAILURE_REPORT.json"
sha256sum "${RUN_ROOT}/PARENT_V9_FAILURE_REPORT.json" > "${RUN_ROOT}/PARENT_V9_FAILURE_REPORT.sha256"

(
  cd "${RUN_ROOT}"
  find data legacy_snapshot -type f -print0 | sort -z | xargs -0 sha256sum
) > "${RUN_ROOT}/status/reused_data_files.sha256"
REUSED_DATA_SHA="$(sha256sum "${RUN_ROOT}/status/reused_data_files.sha256" | cut -d' ' -f1)"
SOURCE_DELTA_SHA="$(sha256sum "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" | cut -d' ' -f1)"
export REUSED_DATA_SHA SOURCE_DELTA_SHA
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema": "h1_chemistry_first_v10_preparation_v1", "status": "pass",
    "source_inventory_sha256": os.environ["SOURCE_SHA256"],
    "source_archive_sha256": os.environ["ARCHIVE_SHA256"],
    "source_delta_audit_sha256": os.environ["SOURCE_DELTA_SHA"],
    "reused_data_tree_manifest_sha256": os.environ["REUSED_DATA_SHA"],
    "repair_scope": "cancellation evidence parser only",
    "training_partition": "gpu", "planner_generation_partition": "gpu",
    "restart_from_protected_p0": True, "resume_cancelled_state": False,
    "broad_tests_repeated": False, "optimizer_smoke_repeated": False,
    "training_math_or_science_change": False,
    "training": False, "generation": False, "smact4_executed_on_a800": False,
}
(root / "V10_PREPARATION_RECORD.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "${RUN_ROOT}/V10_PREPARATION_RECORD.json" > "${RUN_ROOT}/V10_PREPARATION_RECORD.sha256"
touch "${RUN_ROOT}/status/data_SUCCESS" "${RUN_ROOT}/status/source_delta_SUCCESS" \
  "${RUN_ROOT}/status/optimizer_smoke_reuse_SUCCESS" \
  "${RUN_ROOT}/status/optimizer_smoke_sft_v2_SUCCESS" "${RUN_ROOT}/status/optimizer_smoke_sft_v2_c_SUCCESS"
printf '%s\n' pass > "${RUN_ROOT}/status/a800_source_audit.status"
find "${RUN_ROOT}/source" "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" \
  "${RUN_ROOT}/optimizer_smoke" "${ISOLATED_ROOT}" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/source" "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" \
  "${RUN_ROOT}/optimizer_smoke" "${ISOLATED_ROOT}" -type d -exec chmod 500 {} +
find "${STAGING_ROOT}" "${FREEZE_ROOT}" "${TRANSFER_ROOT}" -type f -exec chmod 400 {} +
find "${STAGING_ROOT}" "${FREEZE_ROOT}" "${TRANSFER_ROOT}" -type d -exec chmod 500 {} +
cat "${RUN_ROOT}/V10_PREPARATION_RECORD.json"

#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_RUN="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7"
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected repaired source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected repaired source archive digest}"
EXPECTED_PARENT_DATA_AUDIT_SHA256=1c9a5e2cba51a1258acf107255ca308c7c9f7122d50f5ae3a09ed97d2681612a
EXPECTED_PARENT_ORDER_LEDGER_SHA256=c31df9dd44bbe1ea75a99131899d4e6ea7131d1f230b39ecfb25f730d5406a43
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -x "${LEGACY_PYTHON}"
test -d "${PARENT_RUN}/data"
test -d "${PARENT_RUN}/legacy_snapshot"
test -f "${PARENT_RUN}/status/data_SUCCESS"
test "$(sha256sum "${PARENT_RUN}/data/sft/audit_report.json" | cut -d' ' -f1)" = "${EXPECTED_PARENT_DATA_AUDIT_SHA256}"
test "$(sha256sum "${PARENT_RUN}/data/sft/ORDER_LEDGER.json" | cut -d' ' -f1)" = "${EXPECTED_PARENT_ORDER_LEDGER_SHA256}"
test -d "${RUN_ROOT}/source"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test "$(sha256sum "${RUN_ROOT}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
for path in data legacy_snapshot DATA_REUSE_RECORD.json DATA_REUSE_RECORD.sha256; do
  test ! -e "${RUN_ROOT}/${path}"
done

sacct -n -X -j 31025,31035 -o JobIDRaw,State,ExitCode -P > "${RUN_ROOT}/status/sacct_parent_data_reuse.txt"
test "$(awk -F'|' '$1 == "31025" {print $2 "|" $3}' "${RUN_ROOT}/status/sacct_parent_data_reuse.txt")" = "COMPLETED|0:0"
test "$(awk -F'|' '$1 == "31035" {print $2 "|" $3}' "${RUN_ROOT}/status/sacct_parent_data_reuse.txt")" = "COMPLETED|0:0"

cp -a "${PARENT_RUN}/data" "${RUN_ROOT}/data"
cp -a "${PARENT_RUN}/legacy_snapshot" "${RUN_ROOT}/legacy_snapshot"
cp "${PARENT_RUN}/snapshot_submission_record.json" "${RUN_ROOT}/snapshot_submission_record.json"
sha256sum "${RUN_ROOT}/snapshot_submission_record.json" > "${RUN_ROOT}/snapshot_submission_record.sha256"
cp "${PARENT_RUN}/status/submitted_snapshot_job_id.txt" "${RUN_ROOT}/status/submitted_snapshot_job_id.txt"
cp "${PARENT_RUN}/status/snapshot_SUCCESS" "${RUN_ROOT}/status/snapshot_SUCCESS"
printf '%s\n' 31035 > "${RUN_ROOT}/status/submitted_data_job_id.txt"

(
  cd "${RUN_ROOT}"
  find data legacy_snapshot -type f -print0 | sort -z | xargs -0 sha256sum
) > "${RUN_ROOT}/status/reused_data_files.sha256"
REUSED_TREE_SHA="$(sha256sum "${RUN_ROOT}/status/reused_data_files.sha256" | cut -d' ' -f1)"
export PARENT_RUN RUN_ROOT EXPECTED_SOURCE_INVENTORY_SHA256 EXPECTED_ARCHIVE_SHA256
export EXPECTED_PARENT_DATA_AUDIT_SHA256 EXPECTED_PARENT_ORDER_LEDGER_SHA256 REUSED_TREE_SHA
"${LEGACY_PYTHON}" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema": "h1_chemistry_first_parent_data_reuse_v1",
    "status": "pass",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "parent_run_root": os.environ["PARENT_RUN"],
    "parent_snapshot_job_id": "31025",
    "parent_data_job_id": "31035",
    "parent_data_audit_sha256": os.environ["EXPECTED_PARENT_DATA_AUDIT_SHA256"],
    "parent_order_ledger_sha256": os.environ["EXPECTED_PARENT_ORDER_LEDGER_SHA256"],
    "reused_data_tree_manifest_sha256": os.environ["REUSED_TREE_SHA"],
    "repaired_source_inventory_sha256": os.environ["EXPECTED_SOURCE_INVENTORY_SHA256"],
    "repaired_source_archive_sha256": os.environ["EXPECTED_ARCHIVE_SHA256"],
    "byte_identical_reuse_only": True,
    "data_regenerated": False,
    "smact4_execution_on_a800": False,
    "model_or_optimizer_execution": False,
}
(root / "DATA_REUSE_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
sha256sum "${RUN_ROOT}/DATA_REUSE_RECORD.json" > "${RUN_ROOT}/DATA_REUSE_RECORD.sha256"
find "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" -type d -exec chmod 500 {} +
touch "${RUN_ROOT}/status/data_SUCCESS"
cat "${RUN_ROOT}/DATA_REUSE_RECORD.json"

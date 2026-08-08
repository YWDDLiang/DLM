#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_RUN="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_slurm_array_jobid_repair_v7"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8"
EXPECTED_SOURCE_INVENTORY_SHA256="${1:?expected repaired source inventory digest}"
EXPECTED_ARCHIVE_SHA256="${2:?expected repaired source archive digest}"
EXPECTED_PARENT_DATA_AUDIT_SHA256=1c9a5e2cba51a1258acf107255ca308c7c9f7122d50f5ae3a09ed97d2681612a
EXPECTED_PARENT_ORDER_LEDGER_SHA256=c31df9dd44bbe1ea75a99131899d4e6ea7131d1f230b39ecfb25f730d5406a43
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -x "${LEGACY_PYTHON}"
test -d "${PARENT_RUN}/data"
test -d "${PARENT_RUN}/legacy_snapshot"
test -f "${PARENT_RUN}/status/data_SUCCESS"
test -f "${PARENT_RUN}/status/train_sft_v2_FAILED"
test -f "${PARENT_RUN}/status/train_sft_v2_c_FAILED"
test -f "${PARENT_RUN}/status/planner64_p0_SUCCESS"
test "$(wc -l < "${PARENT_RUN}/planner64/p0/raw_generations.jsonl")" -eq 64
test ! -e "${PARENT_RUN}/planner64/sft_v2/raw_generations.jsonl"
test ! -e "${PARENT_RUN}/planner64/sft_v2_c/raw_generations.jsonl"
test "$(sha256sum "${PARENT_RUN}/data/sft/audit_report.json" | cut -d' ' -f1)" = "${EXPECTED_PARENT_DATA_AUDIT_SHA256}"
test "$(sha256sum "${PARENT_RUN}/data/sft/ORDER_LEDGER.json" | cut -d' ' -f1)" = "${EXPECTED_PARENT_ORDER_LEDGER_SHA256}"
test -d "${RUN_ROOT}/source"
test "$(cat "${RUN_ROOT}/status/a800_source_audit.status")" = pass
test "$(sha256sum "${RUN_ROOT}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)" = "${EXPECTED_ARCHIVE_SHA256}"
for path in data legacy_snapshot DATA_REUSE_RECORD.json DATA_REUSE_RECORD.sha256 V7_TERMINAL_RECORD.json V7_TERMINAL_RECORD.sha256; do
  test ! -e "${RUN_ROOT}/${path}"
done

sacct -n -X -j 31105,31106 -o JobID,State,ExitCode,Elapsed -P \
  > "${RUN_ROOT}/status/sacct_v7_terminal_before_v8.txt"
for expected in \
  '31105_0|FAILED|1:0|00:12:08' \
  '31105_1|FAILED|1:0|00:18:30' \
  '31106_0|COMPLETED|0:0|00:17:21' \
  '31106_1|FAILED|1:0|00:02:46' \
  '31106_2|FAILED|1:0|00:02:08'; do
  grep -Fx "${expected}" "${RUN_ROOT}/status/sacct_v7_terminal_before_v8.txt"
done

cp -a "${PARENT_RUN}/data" "${RUN_ROOT}/data"
cp -a "${PARENT_RUN}/legacy_snapshot" "${RUN_ROOT}/legacy_snapshot"
(
  cd "${RUN_ROOT}"
  find data legacy_snapshot -type f -print0 | sort -z | xargs -0 sha256sum
) > "${RUN_ROOT}/status/reused_data_files.sha256"
REUSED_TREE_SHA="$(sha256sum "${RUN_ROOT}/status/reused_data_files.sha256" | cut -d' ' -f1)"
P0_RAW_SHA="$(sha256sum "${PARENT_RUN}/planner64/p0/raw_generations.jsonl" | cut -d' ' -f1)"
SACCT_SHA="$(sha256sum "${RUN_ROOT}/status/sacct_v7_terminal_before_v8.txt" | cut -d' ' -f1)"
export PARENT_RUN RUN_ROOT EXPECTED_SOURCE_INVENTORY_SHA256 EXPECTED_ARCHIVE_SHA256
export EXPECTED_PARENT_DATA_AUDIT_SHA256 EXPECTED_PARENT_ORDER_LEDGER_SHA256
export REUSED_TREE_SHA P0_RAW_SHA SACCT_SHA
"${LEGACY_PYTHON}" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
created = datetime.now(timezone.utc).isoformat()
reuse = {
    "schema": "h1_chemistry_first_v7_data_reuse_for_v8_v1",
    "status": "pass",
    "created_at": created,
    "parent_run_root": os.environ["PARENT_RUN"],
    "parent_data_audit_sha256": os.environ["EXPECTED_PARENT_DATA_AUDIT_SHA256"],
    "parent_order_ledger_sha256": os.environ["EXPECTED_PARENT_ORDER_LEDGER_SHA256"],
    "reused_data_tree_manifest_sha256": os.environ["REUSED_TREE_SHA"],
    "v8_source_inventory_sha256": os.environ["EXPECTED_SOURCE_INVENTORY_SHA256"],
    "v8_source_archive_sha256": os.environ["EXPECTED_ARCHIVE_SHA256"],
    "byte_identical_reuse_only": True,
    "data_regenerated": False,
    "smact4_execution_on_a800": False,
    "model_or_optimizer_execution": False,
}
terminal = {
    "schema": "h1_chemistry_first_v7_terminal_record_v1",
    "status": "engineering_failure",
    "created_at": created,
    "parent_run_root": os.environ["PARENT_RUN"],
    "training": {
        "31105_0": {"candidate": "sft_v2", "state": "FAILED", "exit_code": "1:0", "elapsed": "00:12:08", "valid_checkpoint": False},
        "31105_1": {"candidate": "sft_v2_c", "state": "FAILED", "exit_code": "1:0", "elapsed": "00:18:30", "valid_checkpoint": False},
    },
    "planner64": {
        "31106_0": {"arm": "p0", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:17:21", "raw_attempts": 64, "raw_sha256": os.environ["P0_RAW_SHA"]},
        "31106_1": {"arm": "sft_v2", "state": "FAILED", "exit_code": "1:0", "elapsed": "00:02:46", "raw_attempts": 0},
        "31106_2": {"arm": "sft_v2_c", "state": "FAILED", "exit_code": "1:0", "elapsed": "00:02:08", "raw_attempts": 0},
    },
    "sacct_snapshot_sha256": os.environ["SACCT_SHA"],
    "root_cause": "first optimizer call has lr_used=0 under the frozen 135-step warmup; V7 incorrectly required immediate candidate parameter change",
    "scientific_result_available": False,
    "smact4_executed_on_a800": False,
}
for name, payload in (("DATA_REUSE_RECORD.json", reuse), ("V7_TERMINAL_RECORD.json", terminal)):
    (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
sha256sum "${RUN_ROOT}/DATA_REUSE_RECORD.json" > "${RUN_ROOT}/DATA_REUSE_RECORD.sha256"
sha256sum "${RUN_ROOT}/V7_TERMINAL_RECORD.json" > "${RUN_ROOT}/V7_TERMINAL_RECORD.sha256"
find "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" -type d -exec chmod 500 {} +
touch "${RUN_ROOT}/status/data_SUCCESS"
cat "${RUN_ROOT}/V7_TERMINAL_RECORD.json"

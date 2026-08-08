#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
TRANSFER_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_transfer_input_gpu_partition_override_v9"
STAGING_ROOT="${PROJECT_ROOT}/runs/20260808_evidence_first_source_staging_gpu_partition_override_v9"
FREEZE_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_source_freeze_gpu_partition_override_v9"
PARENT_RUN="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8"
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_override_v9"
LEGACY_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SOURCE_INPUT_ARCHIVE="${1:?source input archive path}"
EXPECTED_SOURCE_INPUT_SHA256="${2:?source input archive SHA256}"
EXPECTED_PARENT_SOURCE_INVENTORY_SHA256=c58af76cd3df4effc52d99ccaab6e3497f11b4114c59ee0f0c28798ae7c1969c
EXPECTED_PARENT_ARCHIVE_SHA256=5225471f3d5f8dc4e7839504bccdd2c353e5b9ed4f1b819dd99f7b88020ff49f
EXPECTED_LEGACY_EVALUATOR_SHA256=ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178
LEGACY_EVALUATOR_ENTRY=crystal_dlm/composition_validity.py
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
MP20_DIR="${PROJECT_ROOT}/reference/crysllmgen/data/mp_20"

test -x "${LEGACY_PYTHON}"
test -d "${TRANSFER_ROOT}"
test -f "${SOURCE_INPUT_ARCHIVE}"
for path in "${STAGING_ROOT}" "${FREEZE_ROOT}" "${RUN_ROOT}"; do
  test ! -e "${path}"
done
test "$(sha256sum "${SOURCE_INPUT_ARCHIVE}" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INPUT_SHA256}"
test "$(tar -xOf "${SOURCE_INPUT_ARCHIVE}" "${LEGACY_EVALUATOR_ENTRY}" | sha256sum | cut -d' ' -f1)" = \
  "${EXPECTED_LEGACY_EVALUATOR_SHA256}"
test "$(sha256sum "${PARENT_RUN}/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${PARENT_RUN}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_PARENT_ARCHIVE_SHA256}"
(
  cd "${PARENT_RUN}"
  sha256sum -c USER_GPU_PARTITION_OVERRIDE_CANCELLATION.sha256
  sha256sum -c optimizer_smoke_submission_record.sha256
)

mkdir "${STAGING_ROOT}"
tar -xzf "${SOURCE_INPUT_ARCHIVE}" -C "${STAGING_ROOT}" --no-same-owner --no-same-permissions
EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1
EXECUTION_DIR="${STAGING_ROOT}/${EXECUTION_REL}"
for required in \
  freeze_source.py GPU_PARTITION_OVERRIDE_V9.json \
  prepare_gpu_partition_override_v9_on_a800.sh train_v9.sbatch \
  planner64_v9.sbatch submit_training64_v9_once.sh; do
  test -f "${EXECUTION_DIR}/${required}"
done

export CUDA_VISIBLE_DEVICES=
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${LEGACY_PYTHON}" "${EXECUTION_DIR}/freeze_source.py" \
  --project-root "${STAGING_ROOT}" --output-root "${FREEZE_ROOT}"
test -f "${FREEZE_ROOT}/FREEZE_RECORD.json"
test -d "${FREEZE_ROOT}/source"
test -f "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_smact_split_v2.tar.gz"

mkdir "${RUN_ROOT}"
mkdir "${RUN_ROOT}/logs" "${RUN_ROOT}/preflight" "${RUN_ROOT}/status"
cp -a "${FREEZE_ROOT}/source" "${RUN_ROOT}/source"
cp "${FREEZE_ROOT}/h1_chemistry_first_sft_v2_smact_split_v2.tar.gz" \
  "${RUN_ROOT}/source_archive.tar.gz"
cp "${FREEZE_ROOT}/FREEZE_RECORD.json" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json"
SOURCE_ROOT="${RUN_ROOT}/source"
FROZEN_EXECUTION_DIR="${SOURCE_ROOT}/${EXECUTION_REL}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
SOURCE_INVENTORY_SHA256="$(sha256sum SOURCE_SHA256.txt | cut -d' ' -f1)"
SOURCE_ARCHIVE_SHA256="$(sha256sum "${RUN_ROOT}/source_archive.tar.gz" | cut -d' ' -f1)"

"${LEGACY_PYTHON}" - \
  "${PARENT_RUN}/source" "${SOURCE_ROOT}" "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

parent, current, output = map(Path, sys.argv[1:])
ignored = {"SOURCE_MANIFEST.json", "SOURCE_SHA256.txt"}
execution = "workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1/"
allowed_added = {
    execution + "GPU_PARTITION_OVERRIDE_V9.json",
    execution + "prepare_gpu_partition_override_v9_on_a800.sh",
    execution + "train_v9.sbatch",
    execution + "planner64_v9.sbatch",
    execution + "submit_training64_v9_once.sh",
}
allowed_changed = {execution + "test_protocol.py"}

def inventory(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel not in ignored:
                result[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

before = inventory(parent)
after = inventory(current)
added = set(after) - set(before)
removed = set(before) - set(after)
changed = {name for name in set(before) & set(after) if before[name] != after[name]}
failures = []
if added != allowed_added: failures.append("unexpected_added_files")
if removed: failures.append("removed_files")
if changed != allowed_changed: failures.append("unexpected_changed_files")
report = {
    "schema": "h1_chemistry_first_v9_source_delta_audit_v1",
    "status": "pass" if not failures else "fail",
    "parent_source": str(parent),
    "current_source": str(current),
    "added": sorted(added),
    "changed": sorted(changed),
    "removed": sorted(removed),
    "partition_only_runtime_change": not failures,
    "training_math_or_science_change": False,
    "failures": failures,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if failures:
    raise SystemExit(f"V9 source delta audit failed: {failures}")
PY
sha256sum "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" > "${RUN_ROOT}/SOURCE_DELTA_AUDIT.sha256"

export PYTHONPATH="${SOURCE_ROOT}"
cd "${SOURCE_ROOT}"
"${LEGACY_PYTHON}" -m unittest \
  workstreams.final_method_development_20260808.execution.h1_chemistry_first_sft_v2_v1.test_protocol \
  > "${RUN_ROOT}/logs/a800_v9_protocol_tests.out" \
  2> "${RUN_ROOT}/logs/a800_v9_protocol_tests.err"

ISOLATED_ROOT="${RUN_ROOT}/isolated_archive_test"
mkdir "${ISOLATED_ROOT}"
tar -xzf "${RUN_ROOT}/source_archive.tar.gz" -C "${ISOLATED_ROOT}" \
  --no-same-owner --no-same-permissions
cd "${ISOLATED_ROOT}"
sha256sum -c SOURCE_SHA256.txt
test "$(sha256sum crystal_dlm/composition_validity.py | cut -d' ' -f1)" = \
  "${EXPECTED_LEGACY_EVALUATOR_SHA256}"

export PYTHONPATH="${SOURCE_ROOT}"
"${LEGACY_PYTHON}" "${FROZEN_EXECUTION_DIR}/preflight.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${FROZEN_EXECUTION_DIR}/CONFIG.json" \
  --authorization "${FROZEN_EXECUTION_DIR}/AUTHORIZATION.json" \
  --ledger64 "${FROZEN_EXECUTION_DIR}/LEDGER64.json" \
  --ledger256 "${FROZEN_EXECUTION_DIR}/LEDGER256.json" \
  --legacy-python "${LEGACY_PYTHON}" \
  --model-path "${MODEL_PATH}" \
  --p0-adapter-path "${P0_ADAPTER}" \
  --mp20-dir "${MP20_DIR}" \
  --expected-source-inventory-sha256 "${SOURCE_INVENTORY_SHA256}" \
  --output "${RUN_ROOT}/preflight/a800_runtime_preflight.json" \
  > "${RUN_ROOT}/logs/a800_runtime_preflight.out" \
  2> "${RUN_ROOT}/logs/a800_runtime_preflight.err"
sha256sum "${RUN_ROOT}/preflight/a800_runtime_preflight.json" \
  > "${RUN_ROOT}/preflight/a800_runtime_preflight.sha256"

sacct -n -X -j 31136,31137 -o JobID,State,ExitCode,Elapsed -P \
  > "${RUN_ROOT}/status/sacct_parent_v8_cancelled_before_v9.txt"
"${LEGACY_PYTHON}" - "${RUN_ROOT}/status/sacct_parent_v8_cancelled_before_v9.txt" <<'PY'
import sys
expected = {"31136_0", "31136_1", "31137_0", "31137_1", "31137_2"}
rows = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.rstrip("\n").split("|")
    if len(parts) >= 3 and parts[0] in expected:
        rows[parts[0]] = (parts[1], parts[2])
missing = expected - set(rows)
bad = {key: value for key, value in rows.items() if not value[0].startswith("CANCELLED")}
if missing or bad:
    raise SystemExit(f"parent V8 cancellation not terminal: missing={missing}, bad={bad}")
PY
test ! -f "${PARENT_RUN}/status/train_sft_v2_SUCCESS"
test ! -f "${PARENT_RUN}/status/train_sft_v2_c_SUCCESS"

sacct -n -X -j 31126 -o JobID,State,ExitCode,Elapsed -P \
  > "${RUN_ROOT}/status/sacct_parent_v8_optimizer_smoke_before_v9.txt"
for task in 0 1; do
  expected_id="31126_${task}"
  matches="$(awk -F'|' -v wanted="${expected_id}" '$1 == wanted {print $2 "|" $3}' \
    "${RUN_ROOT}/status/sacct_parent_v8_optimizer_smoke_before_v9.txt")"
  test "${matches}" = "COMPLETED|0:0"
done

for path in data legacy_snapshot optimizer_smoke; do
  test -d "${PARENT_RUN}/${path}"
  test ! -e "${RUN_ROOT}/${path}"
  cp -a "${PARENT_RUN}/${path}" "${RUN_ROOT}/${path}"
done
for candidate in sft_v2 sft_v2_c; do
  parent_report="${PARENT_RUN}/optimizer_smoke/${candidate}/optimizer_smoke_report.json"
  report="${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.json"
  test "$(sha256sum "${parent_report}" | cut -d' ' -f1)" = \
    "$(sha256sum "${report}" | cut -d' ' -f1)"
  chmod 600 "${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.sha256"
  sha256sum "${report}" > "${RUN_ROOT}/optimizer_smoke/${candidate}/optimizer_smoke_report.sha256"
done

cp "${PARENT_RUN}/optimizer_smoke_submission_record.json" \
  "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json"
sha256sum "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json" \
  > "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.sha256"
cp "${PARENT_RUN}/USER_GPU_PARTITION_OVERRIDE_CANCELLATION.json" \
  "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json"
sha256sum "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json" \
  > "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.sha256"
cp "${PARENT_RUN}/status/submitted_optimizer_smoke_job_id.txt" \
  "${RUN_ROOT}/status/submitted_optimizer_smoke_job_id.txt"

(
  cd "${RUN_ROOT}"
  find data legacy_snapshot -type f -print0 | sort -z | xargs -0 sha256sum
) > "${RUN_ROOT}/status/reused_data_files.sha256"
REUSED_DATA_TREE_SHA256="$(sha256sum "${RUN_ROOT}/status/reused_data_files.sha256" | cut -d' ' -f1)"
SOURCE_DELTA_AUDIT_SHA256="$(sha256sum "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" | cut -d' ' -f1)"
CANCELLATION_RECORD_SHA256="$(sha256sum "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json" | cut -d' ' -f1)"
SMOKE_SFT_V2_SHA256="$(sha256sum "${RUN_ROOT}/optimizer_smoke/sft_v2/optimizer_smoke_report.json" | cut -d' ' -f1)"
SMOKE_SFT_V2_C_SHA256="$(sha256sum "${RUN_ROOT}/optimizer_smoke/sft_v2_c/optimizer_smoke_report.json" | cut -d' ' -f1)"
export PARENT_RUN RUN_ROOT SOURCE_INVENTORY_SHA256 SOURCE_ARCHIVE_SHA256
export REUSED_DATA_TREE_SHA256 SOURCE_DELTA_AUDIT_SHA256 CANCELLATION_RECORD_SHA256
export SMOKE_SFT_V2_SHA256 SMOKE_SFT_V2_C_SHA256
"${LEGACY_PYTHON}" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema": "h1_chemistry_first_v9_gpu_partition_preparation_v1",
    "status": "pass",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "parent_run_root": os.environ["PARENT_RUN"],
    "source_inventory_sha256": os.environ["SOURCE_INVENTORY_SHA256"],
    "source_archive_sha256": os.environ["SOURCE_ARCHIVE_SHA256"],
    "source_delta_audit_sha256": os.environ["SOURCE_DELTA_AUDIT_SHA256"],
    "parent_cancellation_record_sha256": os.environ["CANCELLATION_RECORD_SHA256"],
    "reused_data_tree_manifest_sha256": os.environ["REUSED_DATA_TREE_SHA256"],
    "optimizer_smoke_report_sha256": {
        "sft_v2": os.environ["SMOKE_SFT_V2_SHA256"],
        "sft_v2_c": os.environ["SMOKE_SFT_V2_C_SHA256"],
    },
    "training_partition": "gpu",
    "planner_generation_partition": "gpu",
    "restart_from_protected_p0": True,
    "resume_cancelled_state": False,
    "training_math_or_science_change": False,
    "optimizer_smoke_repeated": False,
    "smact4_executed_on_a800": False,
    "model_training": False,
    "generation": False,
}
(root / "V9_PREPARATION_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
sha256sum "${RUN_ROOT}/V9_PREPARATION_RECORD.json" > "${RUN_ROOT}/V9_PREPARATION_RECORD.sha256"

touch "${RUN_ROOT}/status/data_SUCCESS"
touch "${RUN_ROOT}/status/optimizer_smoke_sft_v2_SUCCESS"
touch "${RUN_ROOT}/status/optimizer_smoke_sft_v2_c_SUCCESS"
printf '%s\n' pass > "${RUN_ROOT}/status/a800_source_audit.status"
touch "${RUN_ROOT}/status/source_delta_SUCCESS"
touch "${RUN_ROOT}/status/optimizer_smoke_reuse_SUCCESS"
sha256sum "${SOURCE_INPUT_ARCHIVE}" > "${RUN_ROOT}/status/transfer_inputs.sha256"
sha256sum "${RUN_ROOT}/source_archive.tar.gz" > "${RUN_ROOT}/status/source_archive.sha256"
sha256sum "${RUN_ROOT}/source/SOURCE_SHA256.txt" > "${RUN_ROOT}/status/source_inventory.sha256"

find "${RUN_ROOT}/source" "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" \
  "${RUN_ROOT}/optimizer_smoke" "${ISOLATED_ROOT}" -type f -exec chmod 400 {} +
find "${RUN_ROOT}/source" "${RUN_ROOT}/data" "${RUN_ROOT}/legacy_snapshot" \
  "${RUN_ROOT}/optimizer_smoke" "${ISOLATED_ROOT}" -type d -exec chmod 500 {} +
chmod 400 "${RUN_ROOT}/source_archive.tar.gz" "${RUN_ROOT}/SOURCE_FREEZE_RECORD.json" \
  "${RUN_ROOT}/SOURCE_DELTA_AUDIT.json" "${RUN_ROOT}/SOURCE_DELTA_AUDIT.sha256" \
  "${RUN_ROOT}/V9_PREPARATION_RECORD.json" "${RUN_ROOT}/V9_PREPARATION_RECORD.sha256" \
  "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.json" \
  "${RUN_ROOT}/PARENT_V8_OPTIMIZER_SMOKE_SUBMISSION_RECORD.sha256" \
  "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.json" \
  "${RUN_ROOT}/PARENT_V8_USER_CANCELLATION_RECORD.sha256"
find "${STAGING_ROOT}" -type f -exec chmod 400 {} +
find "${STAGING_ROOT}" -type d -exec chmod 500 {} +
find "${FREEZE_ROOT}" -type f -exec chmod 400 {} +
find "${FREEZE_ROOT}" -type d -exec chmod 500 {} +
find "${TRANSFER_ROOT}" -type f -exec chmod 400 {} +
find "${TRANSFER_ROOT}" -type d -exec chmod 500 {} +
cat "${RUN_ROOT}/V9_PREPARATION_RECORD.json"

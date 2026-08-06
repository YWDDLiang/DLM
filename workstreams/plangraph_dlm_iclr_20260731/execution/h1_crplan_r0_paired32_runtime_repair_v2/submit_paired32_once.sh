#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260804_h1_crplan_r0_paired32_runtime_repair_v2"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_r0_paired32_runtime_repair_v2"
ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final/adapter_model.safetensors"
EXPECTED_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
CR0_SUBMISSION="${RUN_ROOT}/status/cr0_submission_record.json"
PAIRED_SUBMISSION="${RUN_ROOT}/status/paired32_submission_record.json"

test -d "${SOURCE_ROOT}"
test -f "${SOURCE_ROOT}/SOURCE_SHA256.txt"
test -f "${EXECUTION_DIR}/paired32.sbatch"
test -f "${ADAPTER}"
test -f "${CR0_SUBMISSION}"
test -f "${RUN_ROOT}/cr0/_SUCCESS"
test -f "${RUN_ROOT}/cr0/terminal_report.json"
test ! -e "${PAIRED_SUBMISSION}"
test ! -e "${RUN_ROOT}/paired32"

partitions="$(sinfo -h -o '%P' | sed 's/*$//' | sort -u)"
grep -qx gpu <<<"${partitions}"

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
SOURCE_MANIFEST_SHA256="$(sha256sum SOURCE_SHA256.txt | cut -d' ' -f1)"
test "$(sha256sum "${ADAPTER}" | cut -d' ' -f1)" = "${EXPECTED_ADAPTER_SHA256}"

CR0_JOB_ID="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["cr0_job_id"])' \
    "${CR0_SUBMISSION}"
)"
RECORDED_SOURCE_SHA256="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["source_manifest_sha256"])' \
    "${CR0_SUBMISSION}"
)"
test "${RECORDED_SOURCE_SHA256}" = "${SOURCE_MANIFEST_SHA256}"
test "$(
  sacct -X -n -P -j "${CR0_JOB_ID}" --format=State,ExitCode |
    sed '/^[[:space:]]*$/d' |
    head -n 1
)" = "COMPLETED|0:0"
"${PYTHON}" -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "pass"' \
  "${RUN_ROOT}/cr0/terminal_report.json"

PAIRED32_JOB_ID="$(
  sbatch --parsable \
    --export=ALL,EXPECTED_SOURCE_MANIFEST_SHA256="${SOURCE_MANIFEST_SHA256}",EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256}" \
    "${EXECUTION_DIR}/paired32.sbatch"
)"

export CR0_JOB_ID PAIRED32_JOB_ID SOURCE_MANIFEST_SHA256
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(
    "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/"
    "runs/20260804_h1_crplan_r0_paired32_runtime_repair_v2/status/"
    "paired32_submission_record.json"
)
payload = {
    "schema": "h1_crplan_r0_paired32_submission_record_v2",
    "status": "paired32_submitted_after_observed_clean_cr0",
    "cr0_job_id": os.environ["CR0_JOB_ID"],
    "paired32_job_id": os.environ["PAIRED32_JOB_ID"],
    "source_manifest_sha256": os.environ["SOURCE_MANIFEST_SHA256"],
    "adapter_model_sha256": (
        "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a"
    ),
    "cr0_observed_state": "COMPLETED",
    "cr0_observed_exit_code": "0:0",
    "cr0_terminal_status": "pass",
    "partitions_preflight": ["gpu"],
    "automatic_four_arm_512": False,
    "automatic_downstream": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf 'PAIRED32_JOB_ID=%s\n' "${PAIRED32_JOB_ID}"

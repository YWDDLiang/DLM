#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
V1_RUN_ROOT="${PROJECT_ROOT}/runs/20260731_plangraph_dlm_g1_v1"
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_plangraph_dlm_g1_sampling_repair_v2"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/g1_source_v2"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
MANIFEST="${EXECUTION_ROOT}/G1_EXECUTION_MANIFEST_V2.json"
AUTHORIZATION="${EXECUTION_ROOT}/G1_AUTHORIZATION_V2.json"
RECORD="${RUN_ROOT}/submission_record.json"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"
test ! -e "${RUN_ROOT}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
test -s "${V1_RUN_ROOT}/data/ledger_report.json"
test -s "${V1_RUN_ROOT}/data/ledger/seed_ledger.jsonl"
test -s "${V1_RUN_ROOT}/training/PG/training_report.json"
test -s "${V1_RUN_ROOT}/training/PG-shuffle/training_report.json"

mkdir -p \
  "${RUN_ROOT}/slurm" \
  "${RUN_ROOT}/status" \
  "${RUN_ROOT}/submission_claim" \
  "${RUN_ROOT}/training/PG" \
  "${RUN_ROOT}/training/PG-shuffle"
cp -a "${V1_RUN_ROOT}/data" "${RUN_ROOT}/data"
cp -a \
  "${V1_RUN_ROOT}/training/PG/training_report.json" \
  "${RUN_ROOT}/training/PG/training_report.json"
cp -a \
  "${V1_RUN_ROOT}/training/PG-shuffle/training_report.json" \
  "${RUN_ROOT}/training/PG-shuffle/training_report.json"
trap 'status=$?; if [ "${status}" -ne 0 ]; then printf "{\"status\":\"failed_before_complete\",\"exit_status\":%s}\\n" "${status}" > "${RECORD}"; fi; exit "${status}"' EXIT

sample_job="$(sbatch --parsable \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/sample.sbatch")"
case "${sample_job}" in *[!0-9]*|'') echo "invalid sample job ${sample_job}" >&2; exit 3;; esac
assembly_job="$(sbatch --parsable \
  --dependency=afterok:"${sample_job}" \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/assemble.sbatch")"
case "${assembly_job}" in *[!0-9]*|'') echo "invalid assembly job ${assembly_job}" >&2; exit 3;; esac

"${PYTHON}" - "${RECORD}" "${sample_job}" "${assembly_job}" "${G1_EXECUTION_MANIFEST_SHA256}" "${G1_AUTHORIZATION_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

path, sample, assembly, manifest_sha, authorization_sha = sys.argv[1:]
payload = {
    "schema": "plangraph-dlm-g1-repair-submission@1",
    "status": "complete",
    "submission_count": 1,
    "repair_cycle": 1,
    "reused_ledger_job": 29098,
    "reused_training_array_job": 29099,
    "superseded_sampling_array_job": 29100,
    "superseded_assembly_job": 29101,
    "sampling_array_job": int(sample),
    "assembly_job": int(assembly),
    "execution_manifest_sha256": manifest_sha,
    "authorization_sha256": authorization_sha,
    "partial_samples_merged": False,
    "all_sampling_arms_rerun_from_ordinal_zero": True,
    "automatic_downstream_within_authorized_chain": True,
    "automatic_G4": False,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

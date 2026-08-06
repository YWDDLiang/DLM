#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260731_plangraph_dlm_g1_v1"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/g1_source_v1"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
MANIFEST="${EXECUTION_ROOT}/G1_EXECUTION_MANIFEST_V1.json"
AUTHORIZATION="${EXECUTION_ROOT}/G1_AUTHORIZATION_V1.json"
RECORD="${RUN_ROOT}/submission_record.json"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"
test ! -e "${RUN_ROOT}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)

mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/status" "${RUN_ROOT}/submission_claim"
trap 'status=$?; if [ "${status}" -ne 0 ]; then printf "{\"status\":\"failed_before_complete\",\"exit_status\":%s}\\n" "${status}" > "${RECORD}"; fi; exit "${status}"' EXIT

ledger_job="$(sbatch --parsable \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/ledger.sbatch")"
case "${ledger_job}" in *[!0-9]*|'') echo "invalid ledger job ${ledger_job}" >&2; exit 3;; esac
train_job="$(sbatch --parsable \
  --dependency=afterok:"${ledger_job}" \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/train.sbatch")"
case "${train_job}" in *[!0-9]*|'') echo "invalid train job ${train_job}" >&2; exit 3;; esac
sample_job="$(sbatch --parsable \
  --dependency=afterok:"${train_job}" \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/sample.sbatch")"
case "${sample_job}" in *[!0-9]*|'') echo "invalid sample job ${sample_job}" >&2; exit 3;; esac
assembly_job="$(sbatch --parsable \
  --dependency=afterok:"${sample_job}" \
  --export=ALL,G1_EXECUTION_MANIFEST_SHA256="${G1_EXECUTION_MANIFEST_SHA256}",G1_AUTHORIZATION_SHA256="${G1_AUTHORIZATION_SHA256}" \
  "${SOURCE_ROOT}/slurm/assemble.sbatch")"
case "${assembly_job}" in *[!0-9]*|'') echo "invalid assembly job ${assembly_job}" >&2; exit 3;; esac

"${PYTHON}" - "${RECORD}" "${ledger_job}" "${train_job}" "${sample_job}" "${assembly_job}" "${G1_EXECUTION_MANIFEST_SHA256}" "${G1_AUTHORIZATION_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

path, ledger, train, sample, assembly, manifest_sha, authorization_sha = sys.argv[1:]
payload = {
    "schema": "plangraph-dlm-g1-submission@1",
    "status": "complete",
    "submission_count": 1,
    "ledger_job": int(ledger),
    "training_array_job": int(train),
    "sampling_array_job": int(sample),
    "assembly_job": int(assembly),
    "execution_manifest_sha256": manifest_sha,
    "authorization_sha256": authorization_sha,
    "automatic_downstream_within_authorized_chain": True,
    "automatic_G4": False,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

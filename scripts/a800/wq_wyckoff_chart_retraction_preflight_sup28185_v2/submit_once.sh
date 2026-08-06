#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
JOB=scripts/a800/wq_wyckoff_chart_retraction_preflight_sup28185_v2/preflight.sbatch
JOB_NAME=wq-chart-ret-wtb32-v2
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
CONTRACT_SHA256=518e44cc1a94334f8232ee54f4199a3c01436c0768defb9e61e2628a27324a6a
RUNNER=scripts/a800/run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py
RUNNER_SHA256=63925aa0b877914b35240e55459026cf10e660665a14304a427c20253ef57a35
LEGACY_RUNNER=scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py
LEGACY_RUNNER_SHA256=edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3
TANGENT=crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py
TANGENT_SHA256=127e3c707b1bf79f2fc44d97bccecd8a6de3cb39b8e989a2806c5c7b377bfbaf
RUNTIME=crystal_dlm/wqcodiff/runtime.py
RUNTIME_SHA256=8b5ba104ee1be25ff7f8a14b703193b33920bfd71abd52b1ba1e0d082e909ea4
AUTHORIZATION=diagnostics/authorization_records/wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
AUTHORIZATION_SHA256=ba1acd31cda69a4388dca420796dd7bee7e7419239ce3b5405668e42bc6e8ce1
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1/generation.jsonl"
SOURCE_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
CHECKPOINT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
CHECKPOINT_SHA256=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
U_AUDIT=runs/remote_audit/20260726_wq_schedule_correct_bridge_parity_sup28054_v1/terminal_audit.json
U_AUDIT_SHA256=4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
U_SELECTION="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/selection_manifest.json"
U_SELECTION_SHA256=547c521c96e3375ffc0665d07c9104d605d14cef2e79c92159339eefb5065cc1
U_ATTEMPTS="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/attempts.jsonl"
U_ATTEMPTS_SHA256=888caaba6683d1b953bf6a7f5a42d9da3d2b432df7a3b1b810b525328b555e94
U_TERMINAL="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/terminal_report.json"
U_TERMINAL_SHA256=730dc0f005676f08d160b37006f1d2d9ababddef5e6ebeb0b200f129e062032e
FAILURE_AUDIT=runs/remote_audit/20260726_wq_wyckoff_tangent_bridge_preflight_v1/terminal_audit_job28185.json
FAILURE_AUDIT_SHA256=2f686b881479f12b4abdc4c0ece217947c5aeb99c072f6119137e97887905f22
RECORD="$RUN/notes/wq_wyckoff_chart_retraction_preflight_sup28185_v2_submission.json"
CLAIM="$RECORD.claim"
OUTPUT="$RUN/outputs/wq_wyckoff_chart_retraction_preflight_sup28185_v2"

: "${WTB32_V2_PATCH_SHA256:?caller must export WTB32_V2_PATCH_SHA256}"
if [[ ! "$WTB32_V2_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid WTB-32 v2 patch identity" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB32_V2_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$LEGACY_RUNNER" | awk '{print $1}')" = "$LEGACY_RUNNER_SHA256"
test "$(sha256sum "$TANGENT" | awk '{print $1}')" = "$TANGENT_SHA256"
test "$(sha256sum "$RUNTIME" | awk '{print $1}')" = "$RUNTIME_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$U_AUDIT" | awk '{print $1}')" = "$U_AUDIT_SHA256"
test "$(sha256sum "$U_SELECTION" | awk '{print $1}')" = "$U_SELECTION_SHA256"
test "$(sha256sum "$U_ATTEMPTS" | awk '{print $1}')" = "$U_ATTEMPTS_SHA256"
test "$(sha256sum "$U_TERMINAL" | awk '{print $1}')" = "$U_TERMINAL_SHA256"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$OUTPUT"

python - "$JOB" <<'PY'
import re
import sys
from pathlib import Path

job = Path(sys.argv[1]).read_text(encoding="utf-8")

def one(pattern: str, name: str) -> str:
    values = re.findall(pattern, job, flags=re.MULTILINE)
    if len(values) != 1:
        raise SystemExit(f"expected exactly one {name} directive")
    return values[0]

partition = one(r"^#SBATCH --partition=(\S+)$", "partition")
cpus = int(one(r"^#SBATCH --cpus-per-task=(\d+)$", "CPU"))
gres = one(r"^#SBATCH --gres=(\S+)$", "GRES")
memory = one(r"^#SBATCH --mem=(\S+)$", "memory")
limit = one(r"^#SBATCH --time=(\S+)$", "time")
match = re.fullmatch(r"gpu:NVIDIAA800-SXM4-80GB:(\d+)", gres)
if not match:
    raise SystemExit("WTB-32 v2 must request exact A800 GRES")
gpus = int(match.group(1))
if (
    partition != "gpu"
    or gpus != 1
    or cpus != 8
    or cpus > 8 * gpus
    or memory != "64G"
    or limit != "01:00:00"
    or "#SBATCH --array" in job
):
    raise SystemExit("WTB-32 v2 resource envelope changed")
PY

existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING \
    -o '%i|%P|%b|%j|%T' | sort -u
)"
if printf '%s\n' "$existing_rows" |
  awk -F '|' -v name="$JOB_NAME" '$4 == name { found=1 } END { exit !found }'; then
  echo "same WTB-32 v2 job identity already exists" >&2
  exit 3
fi

python - "$CLAIM" "$WTB32_V2_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$AUTHORIZATION_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "wq_wyckoff_chart_retraction_submission_claim_v2",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_chart_retraction_preflight_sup28185_v2",
    "supersedes_job_id": 28185,
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "pid": os.getpid(),
    "submit_once": True,
    "evidence_class": "development_mechanics_regression_only",
    "confirmatory_evidence": False,
    "evaluation_only": True,
    "u_rerun": False,
    "training_submitted": False,
    "new_generation_submitted": False,
    "retry_or_replacement_allowed": False,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

command="sbatch --parsable --export=ALL,WTB32_V2_PATCH_SHA256=$WTB32_V2_PATCH_SHA256,WTB32_V2_AUTHORIZATION_SHA256=$AUTHORIZATION_SHA256 $JOB"
job_id="$(
  sbatch --parsable \
    --export="ALL,WTB32_V2_PATCH_SHA256=$WTB32_V2_PATCH_SHA256,WTB32_V2_AUTHORIZATION_SHA256=$AUTHORIZATION_SHA256" \
    "$JOB"
)"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid job ID: $job_id" >&2
  exit 4
fi

python - "$RECORD" "$WTB32_V2_PATCH_SHA256" "$CONTRACT_SHA256" \
  "$AUTHORIZATION_SHA256" "$job_id" "$command" "$existing_rows" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "schema": "wq_wyckoff_chart_retraction_submission_v2",
    "status": "complete",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "identity": "wq_wyckoff_chart_retraction_preflight_sup28185_v2",
    "supersedes_job_id": 28185,
    "execution_patch_sha256": sys.argv[2],
    "scientific_contract_sha256": sys.argv[3],
    "authorization_record_sha256": sys.argv[4],
    "job_id": sys.argv[5],
    "sbatch_command": sys.argv[6],
    "preexisting_queue_rows": [
        row for row in sys.argv[7].splitlines() if row
    ],
    "resources": {
        "partition": "gpu",
        "a800": 1,
        "cpus": 8,
        "memory_gib": 64,
        "time_limit_minutes": 60,
        "cpu_per_a800_gate": "cpus<=8*a800",
    },
    "matrix": {
        "reference_u_cells": 32,
        "u_rerun": False,
        "F_cells": 32,
        "T_cells": 32,
    },
    "evidence_class": "development_mechanics_regression_only",
    "confirmatory_evidence": False,
    "evaluation_only": True,
    "training_submitted": False,
    "new_generation_submitted": False,
    "mlip_submitted": False,
    "external_api_submitted": False,
    "retry_or_replacement_used": False,
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "wq_wyckoff_chart_retraction_preflight_job_id=$job_id"

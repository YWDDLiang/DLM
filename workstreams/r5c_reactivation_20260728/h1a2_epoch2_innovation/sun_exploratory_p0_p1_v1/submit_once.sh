#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1"
RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1"
: "${H1A2C_SUN256_SOURCE_SHA256:?source manifest SHA256 is required}"

if [[ ! "$H1A2C_SUN256_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid source manifest SHA256" >&2
  exit 2
fi

cd "$ROOT"
test ! -e "$RUN"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = \
  "$H1A2C_SUN256_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
for file in ledger generate evaluate assemble; do
  test -s "$SOURCE/slurm/$file.sbatch"
done
test "$(awk -F= '/^#SBATCH --cpus-per-task=/{print $2}' "$SOURCE/slurm/generate.sbatch")" = 8
test "$(awk -F= '/^#SBATCH --cpus-per-task=/{print $2}' "$SOURCE/slurm/evaluate.sbatch")" = 8
test "$(awk -F: '/^#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:/{print $NF}' "$SOURCE/slurm/generate.sbatch")" = 1
test "$(awk -F: '/^#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:/{print $NF}' "$SOURCE/slurm/evaluate.sbatch")" = 1
test "$(awk -F= '/^#SBATCH --array=/{print $2}' "$SOURCE/slurm/generate.sbatch")" = "0-1%2"
test "$(awk -F= '/^#SBATCH --array=/{print $2}' "$SOURCE/slurm/evaluate.sbatch")" = "0-1%2"

mkdir -p "$RUN/logs"
CLAIM="$RUN/submission_claim.json"
RECORD="$RUN/submission_record.json"
python - "$CLAIM" "$H1A2C_SUN256_SOURCE_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "h1a2c_p0_p1_sun256_submission_claim_v1",
    "status": "claimed_before_sbatch",
    "run_id": "20260731_h1a2c_p0_p1_sun256_exploratory_v1",
    "execution_manifest_sha256": sys.argv[2],
    "user_authorized": True,
    "manual_crystal_evaluation_authorized": True,
    "manual_authorization_includes_afterok_sun_evaluation": True,
    "automatic_crystal_evaluation_authorized": False,
    "exploratory_only": True,
    "attempts_per_arm": 256,
    "arms": ["P0", "P1"],
    "submit_once": True,
    "retry_or_replacement_used": False,
    "automatic_downstream_authorized": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

data_job=""
generation_job=""
evaluation_job=""
assembly_job=""
trap 'status=$?; if [[ "$status" -ne 0 && ! -e "$RECORD" ]]; then printf "{\"schema\":\"h1a2c_p0_p1_sun256_submission_v1\",\"status\":\"submission_failed_no_retry\",\"exit_status\":%s,\"data_job\":%s,\"generation_array_job\":%s,\"evaluation_array_job\":%s,\"assembly_job\":%s,\"automatic_downstream_authorized\":false}\\n" "$status" "${data_job:-null}" "${generation_job:-null}" "${evaluation_job:-null}" "${assembly_job:-null}" > "$RECORD"; fi; exit "$status"' EXIT

data_job="$(
  sbatch --parsable \
    --export="ALL,H1A2C_SUN256_SOURCE_SHA256=$H1A2C_SUN256_SOURCE_SHA256" \
    "$SOURCE/slurm/ledger.sbatch"
)"
[[ "$data_job" =~ ^[0-9]+$ ]]
generation_job="$(
  sbatch --parsable \
    --dependency="afterok:$data_job" \
    --export="ALL,H1A2C_SUN256_SOURCE_SHA256=$H1A2C_SUN256_SOURCE_SHA256" \
    "$SOURCE/slurm/generate.sbatch"
)"
[[ "$generation_job" =~ ^[0-9]+$ ]]
evaluation_job="$(
  sbatch --parsable \
    --dependency="afterok:$generation_job" \
    --export="ALL,H1A2C_SUN256_SOURCE_SHA256=$H1A2C_SUN256_SOURCE_SHA256" \
    "$SOURCE/slurm/evaluate.sbatch"
)"
[[ "$evaluation_job" =~ ^[0-9]+$ ]]
assembly_job="$(
  sbatch --parsable \
    --dependency="afterok:$evaluation_job" \
    --export="ALL,H1A2C_SUN256_SOURCE_SHA256=$H1A2C_SUN256_SOURCE_SHA256" \
    "$SOURCE/slurm/assemble.sbatch"
)"
[[ "$assembly_job" =~ ^[0-9]+$ ]]

python - "$RECORD" "$H1A2C_SUN256_SOURCE_SHA256" \
  "$data_job" "$generation_job" "$evaluation_job" "$assembly_job" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "h1a2c_p0_p1_sun256_submission_v1",
    "status": "complete",
    "run_id": "20260731_h1a2c_p0_p1_sun256_exploratory_v1",
    "execution_manifest_sha256": sys.argv[2],
    "jobs": {
        "ledger": sys.argv[3],
        "generation_array": sys.argv[4],
        "evaluation_array": sys.argv[5],
        "assembly": sys.argv[6],
    },
    "dependencies": [
        f"afterok:{sys.argv[3]}",
        f"afterok:{sys.argv[4]}",
        f"afterok:{sys.argv[5]}",
    ],
    "attempts_per_arm": 256,
    "retry_or_replacement_used": False,
    "long_training_submitted": False,
    "manual_crystal_evaluation_authorized": True,
    "manual_authorization_includes_afterok_sun_evaluation": True,
    "automatic_crystal_evaluation_authorized": False,
    "automatic_downstream_authorized": False,
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

printf 'LEDGER_JOB=%s GENERATION_ARRAY_JOB=%s EVALUATION_ARRAY_JOB=%s ASSEMBLY_JOB=%s\n' \
  "$data_job" "$generation_job" "$evaluation_job" "$assembly_job"

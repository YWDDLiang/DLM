#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 RUN_ID command..." >&2
  exit 2
fi

RUN_ID="$1"
shift
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
ENV_NAME="${ENV_NAME:-diff_meets_diff}"
MODEL_ROOT="${MODEL_ROOT:-/public/home/jiaosz/ywliang/models/wqcodiff}"
JOB_NAME="${JOB_NAME:-wqcpu-run}"
CPU_COUNT="${CPU_COUNT:-16}"
SLURM_PARTITION="${SLURM_PARTITION:-short}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
WQ_STAGE="${WQ_STAGE:-unknown}"
MAX_ACTIVE_CPU_JOBS="${MAX_ACTIVE_CPU_JOBS:-4}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ ! "${JOB_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_ID and JOB_NAME may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ "${JOB_NAME}" != wqcpu-* ]]; then
  echo "CPU job names must start with wqcpu- so project concurrency is auditable." >&2
  exit 2
fi
if [ "${WQ_STAGE}" = "unknown" ]; then
  echo "WQ_STAGE must be explicit for every registered CPU job." >&2
  exit 2
fi
if [[ ! "${CPU_COUNT}" =~ ^[0-9]+$ ]] || [ "${CPU_COUNT}" -lt 1 ] || [ "${CPU_COUNT}" -gt 64 ]; then
  echo "CPU_COUNT must be in [1,64]." >&2
  exit 2
fi
if [[ ! "${MAX_ACTIVE_CPU_JOBS}" =~ ^[0-9]+$ ]] || [ "${MAX_ACTIVE_CPU_JOBS}" -lt 1 ]; then
  echo "MAX_ACTIVE_CPU_JOBS must be positive." >&2
  exit 2
fi
case "${SLURM_PARTITION}" in
  short|normal|long) ;;
  *) echo "SLURM_PARTITION must be short, normal, or long." >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/outputs" "runs/${RUN_ID}/slurm" "runs/${RUN_ID}/notes"
SBATCH_PATH="runs/${RUN_ID}/slurm/${JOB_NAME}.sbatch"
if [ -e "${SBATCH_PATH}" ]; then
  echo "Immutable Slurm specification already exists: ${SBATCH_PATH}" >&2
  exit 2
fi

ACTIVE_CPU_JOBS="$(squeue -u "${USER}" -h -t PENDING,RUNNING -o '%j' | awk '/^wqcpu-/ {count++} END {print count + 0}')"
if [ "$((ACTIVE_CPU_JOBS + 1))" -gt "${MAX_ACTIVE_CPU_JOBS}" ]; then
  echo "Submitting would exceed the WQ CPU-job cap: current=${ACTIVE_CPU_JOBS}, cap=${MAX_ACTIVE_CPU_JOBS}." >&2
  exit 2
fi

COMMAND_DECL="COMMAND_ARGV=("
for argument in "$@"; do
  printf -v quoted_argument "%q" "${argument}"
  COMMAND_DECL+=" ${quoted_argument}"
done
COMMAND_DECL+=")"

cat > "${SBATCH_PATH}" <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${SLURM_PARTITION}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_COUNT}
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -o runs/${RUN_ID}/logs/%x-%j.out
#SBATCH -e runs/${RUN_ID}/logs/%x-%j.err

set -Eeuo pipefail
RUN_ID="${RUN_ID}"
PROJECT_ROOT="${PROJECT_ROOT}"
ENV_NAME="${ENV_NAME}"
MODEL_ROOT="${MODEL_ROOT}"
WQ_STAGE="${WQ_STAGE}"
RUN_DIR="\${PROJECT_ROOT}/runs/\${RUN_ID}"
LOG_DIR="\${RUN_DIR}/logs"
OUTPUT_DIR="\${RUN_DIR}/outputs"
FULL_LOG="\${LOG_DIR}/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.full.log"
export RUN_ID PROJECT_ROOT ENV_NAME MODEL_ROOT WQ_STAGE RUN_DIR LOG_DIR OUTPUT_DIR
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="\${RUN_DIR}/.pycache/\${SLURM_JOB_ID}"
mkdir -p "\${LOG_DIR}" "\${OUTPUT_DIR}"
${COMMAND_DECL}

set +e
{
  echo "===== CPU JOB START ====="
  echo "date=\$(date '+%F %T %Z')"
  echo "job_id=\${SLURM_JOB_ID}"
  echo "job_name=\${SLURM_JOB_NAME}"
  echo "nodelist=\${SLURM_JOB_NODELIST}"
  echo "stage=\${WQ_STAGE}"
  echo "cpus=${CPU_COUNT} partition=${SLURM_PARTITION} time=${SLURM_TIME} mem=${SLURM_MEM}"
  cd "\${PROJECT_ROOT}"
  echo "pwd=\$(pwd) host=\$(hostname) user=\$(whoami)"
  set +u
  if [ -f /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
  else
    source ~/.bashrc
  fi
  set -u
  conda activate "\${ENV_NAME}"
  # Never inherit a login-node PYTHONPATH.  An absolute test/script path makes
  # its own directory sys.path[0], so an inherited older runtime can otherwise
  # shadow PROJECT_ROOT even after cd'ing here.
  export PYTHONPATH="\${PROJECT_ROOT}"
  python -V
  python - <<'PY'
import os
from pathlib import Path

import crystal_dlm.wqcodiff as wqcodiff

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
module_path = Path(wqcodiff.__file__).resolve()
if project_root != module_path and project_root not in module_path.parents:
    raise SystemExit(
        f"runtime source mismatch: expected under {project_root}, got {module_path}"
    )
print(f"wqcodiff_runtime={module_path}")
PY
  python scripts/a800/env_doctor.py \
    --project-root "\${PROJECT_ROOT}" \
    --model-root "\${MODEL_ROOT}" \
    --require-slurm --require-offline --skip-assets
  python -m pip freeze | sha256sum || true
  echo "===== COMMAND ====="
  printf '%q ' "\${COMMAND_ARGV[@]}"
  printf '\n'
  started_epoch=\$(date +%s)
  set +e
  /usr/bin/time -v "\${COMMAND_ARGV[@]}"
  command_status=\$?
  set -e
  ended_epoch=\$(date +%s)
  python - "\${OUTPUT_DIR}/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.cpu_usage.json" \
    "\${started_epoch}" "\${ended_epoch}" "${CPU_COUNT}" "\${command_status}" <<'PY'
import json, os, sys
from pathlib import Path
elapsed = max(0, int(sys.argv[3]) - int(sys.argv[2]))
payload = {
    "schema": "wqcodiff_slurm_cpu_usage_v1",
    "run_id": os.environ["RUN_ID"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_name": os.environ["SLURM_JOB_NAME"],
    "stage": os.environ["WQ_STAGE"],
    "elapsed_s": elapsed,
    "allocated_cpu_count": int(sys.argv[4]),
    "cpu_hours_allocated": elapsed * int(sys.argv[4]) / 3600.0,
    "command_status": int(sys.argv[5]),
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  echo "command_status=\${command_status}"
  echo "date=\$(date '+%F %T %Z')"
  echo "===== CPU JOB END ====="
  exit "\${command_status}"
} 2>&1 | tee -a "\${FULL_LOG}"

status=\${PIPESTATUS[0]}
set -e
echo "final_status=\${status}" | tee -a "\${FULL_LOG}"
exit "\${status}"
EOF

echo "Created ${SBATCH_PATH}"
sbatch "${SBATCH_PATH}"

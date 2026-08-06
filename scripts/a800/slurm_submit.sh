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
JOB_NAME="${JOB_NAME:-wqcodiff-run}"
GPU_COUNT="${GPU_COUNT:-1}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-120G}"
SLURM_NODELIST="${SLURM_NODELIST:-}"
WQ_WEEK="${WQ_WEEK:-}"
WQ_METHOD="${WQ_METHOD:-unknown}"
WQ_STAGE="${WQ_STAGE:-unknown}"
WQ_RUNTIME_SCOPE="${WQ_RUNTIME_SCOPE:-core}"
PROPOSED_GPU_HOURS="${PROPOSED_GPU_HOURS:-}"
GPU_BUDGET_RUNS_ROOTS="${GPU_BUDGET_RUNS_ROOTS:-runs}"

if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [ "${GPU_COUNT}" -lt 1 ] || [ "${GPU_COUNT}" -gt 2 ]; then
  echo "GPU_COUNT must be in [1,2] for this project." >&2
  exit 2
fi
if [ "${GPU_COUNT}" -gt 1 ] && [ "${ALLOW_MULTI_GPU_JOB:-0}" != "1" ]; then
  echo "Default policy is one model per GPU. Set ALLOW_MULTI_GPU_JOB=1 only after the P0 timing audit." >&2
  exit 2
fi
if [[ ! "${WQ_WEEK}" =~ ^[1-4]$ ]]; then
  echo "WQ_WEEK must be one of 1,2,3,4 for GPU-hour accounting." >&2
  exit 2
fi
if [ "${WQ_METHOD}" = "unknown" ] || [ "${WQ_STAGE}" = "unknown" ]; then
  echo "WQ_METHOD and WQ_STAGE must be explicit for every registered job." >&2
  exit 2
fi
case "${WQ_RUNTIME_SCOPE}" in
  core|chgnet|mattersim|mace) ;;
  *) echo "WQ_RUNTIME_SCOPE must be core, chgnet, mattersim, or mace." >&2; exit 2 ;;
esac
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ ! "${JOB_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_ID and JOB_NAME may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ ! "${PROPOSED_GPU_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PROPOSED_GPU_HOURS is required for the pre-submission budget gate." >&2
  exit 2
fi
if [ "${SLURM_PARTITION}" != "gpu" ]; then
  echo "SLURM_PARTITION must be gpu; gpu_long is disabled by execution policy." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/outputs" "runs/${RUN_ID}/slurm" "runs/${RUN_ID}/notes"
SBATCH_PATH="runs/${RUN_ID}/slurm/${JOB_NAME}.sbatch"
if [ -e "${SBATCH_PATH}" ]; then
  echo "Immutable Slurm specification already exists: ${SBATCH_PATH}" >&2
  exit 2
fi

CURRENT_GPUS="$(squeue -u "${USER}" -h -t PENDING,RUNNING -o '%b' | awk -F: '
  /gpu:/ {
    value=$NF
    sub(/[^0-9].*$/, "", value)
    if (value ~ /^[0-9]+$/) total += value
  }
  END { print total + 0 }
')"
if [ "$((CURRENT_GPUS + GPU_COUNT))" -gt 2 ]; then
  echo "Submitting would exceed the two-GPU project concurrency cap: current=${CURRENT_GPUS}, requested=${GPU_COUNT}." >&2
  exit 2
fi

IFS=':' read -r -a budget_roots <<< "${GPU_BUDGET_RUNS_ROOTS}"
budget_root_args=()
for root in "${budget_roots[@]}"; do
  if [ -z "${root}" ]; then
    echo "GPU_BUDGET_RUNS_ROOTS contains an empty path." >&2
    exit 2
  fi
  budget_root_args+=(--runs-root "${root}")
done
python scripts/a800/gpu_budget.py \
  "${budget_root_args[@]}" \
  --current-week "${WQ_WEEK}" \
  --proposed-gpu-hours "${PROPOSED_GPU_HOURS}" \
  --output "runs/${RUN_ID}/notes/${JOB_NAME}.budget.json"

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
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:${GPU_COUNT}
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=${SLURM_MEM}
$(if [ -n "${SLURM_NODELIST}" ]; then echo "#SBATCH --nodelist=${SLURM_NODELIST}"; fi)
#SBATCH -o runs/${RUN_ID}/logs/%x-%j.out
#SBATCH -e runs/${RUN_ID}/logs/%x-%j.err

set -Eeuo pipefail

RUN_ID="${RUN_ID}"
PROJECT_ROOT="${PROJECT_ROOT}"
ENV_NAME="${ENV_NAME}"
MODEL_ROOT="${MODEL_ROOT}"
GPU_COUNT="${GPU_COUNT}"
WQ_WEEK="${WQ_WEEK}"
WQ_METHOD="${WQ_METHOD}"
WQ_STAGE="${WQ_STAGE}"
WQ_RUNTIME_SCOPE="${WQ_RUNTIME_SCOPE}"
RUN_DIR="\${PROJECT_ROOT}/runs/\${RUN_ID}"
LOG_DIR="\${RUN_DIR}/logs"
OUTPUT_DIR="\${RUN_DIR}/outputs"
FULL_LOG="\${LOG_DIR}/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.full.log"
export RUN_ID PROJECT_ROOT ENV_NAME MODEL_ROOT GPU_COUNT WQ_WEEK WQ_METHOD WQ_STAGE WQ_RUNTIME_SCOPE RUN_DIR LOG_DIR OUTPUT_DIR
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
  echo "===== JOB START ====="
  echo "date=\$(date '+%F %T %Z')"
  echo "job_id=\${SLURM_JOB_ID}"
  echo "job_name=\${SLURM_JOB_NAME}"
  echo "nodelist=\${SLURM_JOB_NODELIST}"
  echo "project_root=\${PROJECT_ROOT}"
  echo "run_dir=\${RUN_DIR}"
  echo "gpu_count=${GPU_COUNT}"
  echo "runtime_scope=${WQ_RUNTIME_SCOPE}"
  echo "partition=${SLURM_PARTITION}"
  echo "slurm_time=${SLURM_TIME}"
  echo "slurm_mem=${SLURM_MEM}"
  echo
  cd "\${PROJECT_ROOT}"
  echo "pwd=\$(pwd)"
  echo "hostname=\$(hostname)"
  echo "whoami=\$(whoami)"
  echo
  echo "===== NVIDIA-SMI BEFORE ====="
  nvidia-smi || true
  echo
  echo "===== ENVIRONMENT ====="
  set +u
  if [ -f /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
  else
    source ~/.bashrc
  fi
  set -u
  conda activate "\${ENV_NAME}"
  # Drop any login-node runtime from PYTHONPATH before selecting this job's
  # frozen source tree.  Appending the inherited value can silently import an
  # older wqcodiff package when the command is an absolute script path.
  export PYTHONPATH="\${PROJECT_ROOT}"
  if [ "\${WQ_RUNTIME_SCOPE}" = "mattersim" ]; then
    export PYTHONPATH="\${MODEL_ROOT}/runtimes/mattersim-1.1.2-py310-v4:\${PROJECT_ROOT}"
    export PYTHONDONTWRITEBYTECODE=1
  fi
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
  python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required inside a registered Slurm GPU job")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY
  doctor_args=( \
    --project-root "\${PROJECT_ROOT}" \
    --model-root "\${MODEL_ROOT}" \
    --runtime-scope "\${WQ_RUNTIME_SCOPE}" \
    --require-cuda --require-slurm --require-offline \
  )
  if [ "\${WQ_RUNTIME_SCOPE}" = "core" ]; then
    doctor_args+=(--skip-assets)
  fi
  python scripts/a800/env_doctor.py "\${doctor_args[@]}"
  python -m pip freeze | sha256sum || true
  echo
  echo "===== COMMAND ====="
  printf '%q ' "\${COMMAND_ARGV[@]}"
  printf '\n'
  usage_csv="\${LOG_DIR}/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.gpu.csv"
  started_epoch=\$(date +%s)
  nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
    --format=csv,noheader,nounits -l 5 > "\${usage_csv}" &
  monitor_pid=\$!
  set +e
  "\${COMMAND_ARGV[@]}"
  command_status=\$?
  set -e
  kill "\${monitor_pid}" 2>/dev/null || true
  wait "\${monitor_pid}" 2>/dev/null || true
  ended_epoch=\$(date +%s)
  python - "\${usage_csv}" "\${OUTPUT_DIR}/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.job_usage.json" \
    "\${started_epoch}" "\${ended_epoch}" "\${GPU_COUNT}" <<'PY'
import csv, json, os, sys
from pathlib import Path
rows = []
path = Path(sys.argv[1])
if path.exists():
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 5:
                rows.append({
                    "timestamp": row[0].strip(),
                    "gpu_index": int(row[1]),
                    "memory_used_mib": float(row[2]),
                    "power_w": float(row[3]),
                    "utilization_percent": float(row[4]),
                })
elapsed = max(0, int(sys.argv[4]) - int(sys.argv[3]))
gpu_count = int(sys.argv[5])
payload = {
    "schema": "wqcodiff_slurm_usage_v1",
    "run_id": os.environ["RUN_ID"],
    "week": int(os.environ["WQ_WEEK"]),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_name": os.environ["SLURM_JOB_NAME"],
    "method": os.environ["WQ_METHOD"],
    "stage": os.environ["WQ_STAGE"],
    "runtime_scope": os.environ["WQ_RUNTIME_SCOPE"],
    "elapsed_s": elapsed,
    "allocated_gpu_count": gpu_count,
    "gpu_hours": elapsed * gpu_count / 3600.0,
    "peak_memory_mib": max((row["memory_used_mib"] for row in rows), default=None),
    "mean_power_w": (
        sum(row["power_w"] for row in rows) / len(rows) if rows else None
    ),
    "mean_utilization_percent": (
        sum(row["utilization_percent"] for row in rows) / len(rows)
        if rows else None
    ),
    "peak_utilization_percent": max(
        (row["utilization_percent"] for row in rows), default=None
    ),
    "samples": len(rows),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  echo "command_status=\${command_status}"
  echo
  echo "===== NVIDIA-SMI AFTER ====="
  nvidia-smi || true
  echo "date=\$(date '+%F %T %Z')"
  echo "===== JOB END ====="
  exit "\${command_status}"
} 2>&1 | tee -a "\${FULL_LOG}"

status=\${PIPESTATUS[0]}
set -e
echo "final_status=\${status}" | tee -a "\${FULL_LOG}"
exit "\${status}"
EOF

echo "Created ${SBATCH_PATH}"
sbatch "${SBATCH_PATH}"

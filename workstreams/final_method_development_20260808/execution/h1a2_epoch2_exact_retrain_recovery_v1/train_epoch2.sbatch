#!/bin/bash
#SBATCH -J h1a2-rtrn
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH -o /public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1/logs/%x-%j.out
#SBATCH -e /public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1"
SOURCE="$RUN/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
BASE=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/
EPOCH1=runs/20260602_182700-h1a2-rich-l3base-256/outputs/h1a2_llama_rich_sft/final
DATA=data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base
OUTPUT_REL=runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1/outputs/h1a2_epoch2_llama_rich_sft
OUTPUT="$PROJECT/$OUTPUT_REL"
: "${H1_EXACT_RETRAIN_SOURCE_SHA256:?source manifest SHA is required}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
unset MP_API_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY || true

finish() {
  status=$?
  printf '%s\n' "$status" > "$RUN/status/training_exit_code.txt"
  if [ "$status" -ne 0 ]; then
    : > "$RUN/status/training_FAILED"
  fi
  exit "$status"
}
trap finish EXIT

test -d "$RUN"
test -d "$SOURCE"
test ! -e "$RUN/status/training_STARTED"
: > "$RUN/status/training_STARTED"
cd "$PROJECT"

"$PYTHON" "$SOURCE/preflight.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_EXACT_RETRAIN_SOURCE_SHA256" \
  --phase job \
  --output "$RUN/status/job_preflight_report.json"

"$PYTHON" - "$RUN/status/runtime_versions.json" <<'PY'
import importlib.metadata as m, json, os, platform, socket, sys
path = sys.argv[1]
payload = {
    "python": platform.python_version(),
    "packages": {name: m.version(name) for name in ("accelerate", "numpy", "peft", "safetensors", "tokenizers", "torch", "transformers")},
    "hostname": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_device_count": len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
PY

mkdir -p "$RUN/outputs"
printf 'STAGE exact_h1a2_epoch2_training_start\n'
set +e
"$PYTHON" "$PROJECT/scripts/llama_formula_sft.py" \
  --model-path "$BASE" \
  --checkpoint-path "$EPOCH1" \
  --data-dir "$DATA" \
  --output-dir "$OUTPUT_REL" \
  --max-length 768 \
  --epochs 1.0 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 2e-5 \
  2>&1 | tee "$RUN/logs/exact_training.log"
train_status=${PIPESTATUS[0]}
set -e
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

printf 'STAGE exact_h1a2_epoch2_training_verify\n'
"$PYTHON" "$SOURCE/verify_training.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_EXACT_RETRAIN_SOURCE_SHA256" \
  --training-dir "$OUTPUT" \
  --output "$RUN/training_terminal_report.json"

: > "$RUN/status/training_SUCCESS"
printf 'STAGE exact_h1a2_epoch2_training_complete\n'

#!/bin/bash
#SBATCH --job-name=h1-sun-clean
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_sun_official_gga_u_clean_reeval_v1/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_sun_official_gga_u_clean_reeval_v1/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260811_h1_sun_official_gga_u_clean_reeval_v1"
SOURCE="$RUN/source"
EVAL_PYTHON="/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python"
INDEX="${SLURM_ARRAY_TASK_ID:?array task ID is required}"
: "${H1_SUN_CLEAN_SOURCE_SHA256:?source manifest SHA is required}"
STATUS="$RUN/status/cell_${INDEX}_exit_code.txt"

on_exit() {
  rc=$?
  printf '%s\n' "$rc" > "$STATUS"
  if [[ "$rc" -ne 0 ]]; then
    touch "$RUN/status/cell_${INDEX}_FAILURE"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ ! "$H1_SUN_CLEAN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "resource or source contract changed" >&2
  exit 2
fi
test -f "$RUN/official_mp_cache/completion_SUCCESS"
if compgen -G "$RUN/cells/$(printf '%03d' "$INDEX")_*" > /dev/null; then
  echo "immutable cell output already exists" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

cd "$PROJECT"
test -x "$EVAL_PYTHON"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$EVAL_PYTHON" "$SOURCE/reevaluate_cell.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_SUN_CLEAN_SOURCE_SHA256" \
  --run-root "$RUN" \
  --cell-index "$INDEX"

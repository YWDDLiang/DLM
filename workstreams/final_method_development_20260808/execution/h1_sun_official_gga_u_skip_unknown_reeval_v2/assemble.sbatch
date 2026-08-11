#!/bin/bash
#SBATCH --job-name=h1-sun-skipunk-assemble
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2/logs/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2"
SOURCE="$RUN/source"
EVAL_PYTHON="/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python"
: "${H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256:?source manifest SHA is required}"

on_exit() {
  rc=$?
  printf '%s\n' "$rc" > "$RUN/status/assembly_exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    touch "$RUN/status/assembly_FAILURE"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]]; then
  echo "assembly resource contract changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -x "$EVAL_PYTHON"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
"$EVAL_PYTHON" "$SOURCE/assemble.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
  --run-root "$RUN"

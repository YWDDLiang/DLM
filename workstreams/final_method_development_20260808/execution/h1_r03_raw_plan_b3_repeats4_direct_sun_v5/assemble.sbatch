#!/bin/bash
#SBATCH --job-name=h1-r03p-b3-asm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_r03g_schema_repair_v5/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_r03g_schema_repair_v5/logs/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_r03g_schema_repair_v5"
SOURCE="$RUN_ROOT/source"
: "${H1_R03B3_SOURCE_SHA256:?source manifest SHA256 is required}"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$RUN_ROOT/status/assembly_exit_code.txt"
  if [[ -f "$RUN_ROOT/terminal_report.json" ]]; then
    sha256sum "$RUN_ROOT/terminal_report.json" > "$RUN_ROOT/status/terminal_report.sha256"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_R03B3_SOURCE_SHA256" ]]; then
  echo "R03/B3 assembly resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test ! -e "$RUN_ROOT/terminal_report.json"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

python "$SOURCE/assemble_repeats.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --run-root "$RUN_ROOT" \
  --source-manifest-sha256 "$H1_R03B3_SOURCE_SHA256" \
  --output "$RUN_ROOT/terminal_report.json"

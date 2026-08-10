#!/bin/bash
#SBATCH --job-name=h1-plan1200-assembly
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/body-assembly-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/body-assembly-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/body_source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_PLAN1200_BODY_SOURCE_SHA256:?body source manifest SHA is required}"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$RUN_ROOT/status/body_assembly_exit_code.txt"
  if [[ -f "$RUN_ROOT/terminal_report.json" ]]; then
    sha256sum "$RUN_ROOT/terminal_report.json" > "$RUN_ROOT/status/terminal_report.sha256"
  fi
  if [[ -f "$RUN_ROOT/RESULTS_COMPLETE.md" ]]; then
    sha256sum "$RUN_ROOT/RESULTS_COMPLETE.md" > "$RUN_ROOT/status/results_complete_markdown.sha256"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_PLAN1200_BODY_SOURCE_SHA256" ]]; then
  echo "assembly resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/body_preparation_SUCCESS"
test ! -e "$RUN_ROOT/terminal_report.json"
test ! -e "$RUN_ROOT/RESULTS_COMPLETE.md"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/assembly_${SLURM_JOB_ID}"
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/assemble_panel.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_PLAN1200_BODY_SOURCE_SHA256" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/terminal_report.json" \
  --markdown "$RUN_ROOT/RESULTS_COMPLETE.md"

echo "STAGE body_assembly_complete"

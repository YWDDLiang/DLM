#!/bin/bash
#SBATCH --job-name=h1-native1000-assembly
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/native-assembly-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3/logs/native-assembly-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/native1000_source"
BODY_SOURCE="$RUN_ROOT/body_source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_NATIVE1000_SOURCE_SHA256:?native source manifest SHA is required}"

on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$RUN_ROOT/status/native1000_assembly_exit_code.txt"
  for artifact in \
    "$RUN_ROOT/crysllmgen_native1000/terminal_report.json" \
    "$RUN_ROOT/CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md" \
    "$RUN_ROOT/RESULTS_COMPLETE_WITH_NATIVE1000.md"; do
    if [[ -f "$artifact" ]]; then
      sha256sum "$artifact" > "$RUN_ROOT/status/$(basename "$artifact").sha256"
    fi
  done
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 4 ]] \
  || [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" != "$H1_NATIVE1000_SOURCE_SHA256" ]]; then
  echo "native assembly resource or source identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test -f "$RUN_ROOT/status/native1000_preparation_SUCCESS"
test -f "$RUN_ROOT/status/native1000_submission_record.json"
test ! -e "$RUN_ROOT/crysllmgen_native1000/terminal_report.json"
test ! -e "$RUN_ROOT/CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md"
test ! -e "$RUN_ROOT/RESULTS_COMPLETE_WITH_NATIVE1000.md"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE:$BODY_SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/native1000_assembly_${SLURM_JOB_ID}"
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/assemble_native_panel.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_NATIVE1000_SOURCE_SHA256" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/crysllmgen_native1000/terminal_report.json" \
  --markdown "$RUN_ROOT/CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md" \
  --combined-markdown "$RUN_ROOT/RESULTS_COMPLETE_WITH_NATIVE1000.md"

echo "STAGE native1000_assembly_complete"

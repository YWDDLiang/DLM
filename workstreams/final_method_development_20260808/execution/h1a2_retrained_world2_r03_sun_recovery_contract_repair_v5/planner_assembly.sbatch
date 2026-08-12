#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-pasm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/planner-assembly-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/planner-assembly-%j.err

set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
SOURCE="$RUN_ROOT/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"
EXIT_CODE="$RUN_ROOT/status/planner_assembly_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/planner_assembly_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT
test "${SLURM_JOB_PARTITION:-}" = normal
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
export PYTHONPATH="$SOURCE"
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
"$PYTHON" "$SOURCE/assemble_planners.py" --run-root "$RUN_ROOT" --source-dir "$SOURCE" --source-manifest-sha256 "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
echo "STAGE planner_assembly_complete"

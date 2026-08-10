#!/bin/bash
#SBATCH --job-name=h1-p0-plan1200-asm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2/logs/planner-assembly-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2/logs/planner-assembly-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2"
SOURCE="$RUN_ROOT/planner_source"
EXECUTION="$SOURCE/workstreams/final_method_development_20260808/execution/h1_p0_plan1200_r03_b3_prepost_repeats3_v2"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_PLAN1200_SOURCE_SHA256:?planner source manifest SHA is required}"
: "${H1_PLAN1200_ARRAY_JOB_ID:?planner array job id is required}"

test "${SLURM_JOB_PARTITION:-}" = normal
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_PLAN1200_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

sacct -j "$H1_PLAN1200_ARRAY_JOB_ID" \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList \
  --parsable2 --noheader > "$RUN_ROOT/status/planner_array_sacct.txt" || true

export PYTHONPATH="$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
"$PYTHON" "$EXECUTION/assemble_planners.py" \
  --run-root "$RUN_ROOT" \
  --source-manifest-sha256 "$H1_PLAN1200_SOURCE_SHA256" \
  --output "$RUN_ROOT/planner_terminal_report.json"

echo "STAGE planner_assembly_complete"

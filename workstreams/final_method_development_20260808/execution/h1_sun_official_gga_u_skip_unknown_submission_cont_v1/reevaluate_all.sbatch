#!/bin/bash
#SBATCH --job-name=h1-sun-skipall
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2/logs/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260812_h1_sun_official_gga_u_skip_unknown_reeval_v2"
SOURCE="$RUN/source"
EVAL_PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256:?run-source manifest SHA is required}"

on_exit() {
  rc=$?
  printf '%s\n' "$rc" > "$RUN/status/reevaluate_all_exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    touch "$RUN/status/reevaluate_all_FAILURE"
  fi
  return "$rc"
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != normal ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 16 ]] \
  || [[ ! "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "resource or source contract changed" >&2
  exit 2
fi
test -f "$RUN/official_mp_cache/completion_SUCCESS"
test ! -e "$RUN/status/reevaluate_all_SUCCESS"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

cd "$PROJECT"
test -x "$EVAL_PYTHON"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
for index in $(seq 0 15); do
  if compgen -G "$RUN/cells/$(printf '%03d' "$index")_*" > /dev/null; then
    echo "immutable cell output already exists for index $index" >&2
    exit 2
  fi
  (
    set +e
    "$EVAL_PYTHON" "$SOURCE/reevaluate_cell.py" \
      --source-dir "$SOURCE" \
      --source-manifest-sha256 "$H1_SUN_SKIP_UNKNOWN_SOURCE_SHA256" \
      --run-root "$RUN" \
      --cell-index "$index" \
      > "$RUN/logs/parallel_cell_${index}.out" \
      2> "$RUN/logs/parallel_cell_${index}.err"
    rc=$?
    printf '%s\n' "$rc" > "$RUN/status/cell_${index}_exit_code.txt"
    if [[ "$rc" -ne 0 ]]; then
      touch "$RUN/status/cell_${index}_FAILURE"
    fi
    exit "$rc"
  ) &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
touch "$RUN/status/reevaluate_all_SUCCESS"


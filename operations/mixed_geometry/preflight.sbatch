#!/usr/bin/env bash
#SBATCH --job-name=dlm-mixed-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=dlm-mixed-preflight-%j.out
#SBATCH --error=dlm-mixed-preflight-%j.err
set -Eeuo pipefail
RUN="${MIXED_ROOT:?}/runs/preflight_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${MIXED_CODE_COMMIT:?}")" = "$MIXED_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$MIXED_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXED_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
export PYTHONPATH="$RUN/code/src:$RUN/code/tests" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${MIXED_PYTHON:?}" -m unittest discover -s "$RUN/code/tests" -p 'test_mixed_geometry*.py' -v \
  >"$RUN/cpu-tests.out" 2>"$RUN/cpu-tests.err"
"$MIXED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${MIXED_GPUS:-4}" \
  "$RUN/code/src/scripts/preflight_mixed_geometry_dlm.py" --model-path "${MIXED_MODEL_PATH:?}" \
  --v2-checkpoint "${MIXED_V2_CHECKPOINT:?}" --val-data "${MIXED_VAL_DATA:?}" \
  --val-identity "${MIXED_VAL_IDENTITY:?}" --normalizer "${MIXED_NORMALIZER:?}" \
  --output-dir "$RUN/preflight" --benchmark-updates 32 \
  >"$RUN/preflight.out" 2>"$RUN/preflight.err"
test -f "$RUN/preflight/_SUCCESS"
touch "$RUN/_SUCCESS"

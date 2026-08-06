#!/usr/bin/env bash
set -Eeuo pipefail
set -o noclobber

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_attribution_v1"
RUN="$ROOT/runs/20260803_h1_body_safeaxis_refined_repeats4_attribution_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_R03H_SOURCE_SHA256:?R03H source manifest SHA256 is required}"

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_NAME:-}" ]]; then
  echo "ERROR: R03H is A800 login-node-only" >&2
  exit 2
fi
if [[ ! "$H1_R03H_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: invalid R03H source manifest SHA256" >&2
  exit 3
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: runtime is missing" >&2
  exit 4
fi
if [[ -e "$RUN" ]]; then
  echo "ERROR: fixed R03H run identity already exists" >&2
  exit 5
fi

cd "$ROOT"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = \
  "$H1_R03H_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

umask 077
mkdir -p "$RUN/logs"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/analyze.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --run-root "$RUN" \
  --execution-manifest-sha256 "$H1_R03H_SOURCE_SHA256" \
  >"$RUN/logs/attribution.out" \
  2>"$RUN/logs/attribution.err"

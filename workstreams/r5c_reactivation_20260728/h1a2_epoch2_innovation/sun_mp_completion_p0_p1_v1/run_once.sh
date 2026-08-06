#!/usr/bin/env bash
set -Eeuo pipefail
set -o noclobber

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_mp_completion_p0_p1_v4"
RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_mpcomplete_v4"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1A2C_MP_COMPLETION_SOURCE_SHA256:?completion source manifest SHA256 is required}"
: "${H1A2C_MP_KEY_FILE:?private MP key file path is required}"

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_NAME:-}" ]]; then
  echo "ERROR: MP reference completion is login-node-only" >&2
  exit 2
fi
if [[ ! "$H1A2C_MP_COMPLETION_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: invalid completion source manifest SHA256" >&2
  exit 3
fi
if [[ ! -x "$PYTHON" || ! -s "$H1A2C_MP_KEY_FILE" ]]; then
  echo "ERROR: runtime or private key carrier is missing" >&2
  exit 4
fi
if [[ -e "$RUN" ]]; then
  echo "ERROR: fixed completion run identity already exists" >&2
  exit 5
fi

cd "$ROOT"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = \
  "$H1A2C_MP_COMPLETION_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

umask 077
mkdir -p "$RUN/logs"
trap 'status=$?; if [[ -n "${H1A2C_MP_KEY_FILE:-}" && -f "${H1A2C_MP_KEY_FILE}" ]]; then rm -f "${H1A2C_MP_KEY_FILE}"; fi; exit "$status"' EXIT
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/complete_sun.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --run-root "$RUN" \
  --key-file "$H1A2C_MP_KEY_FILE" \
  --execution-manifest-sha256 "$H1A2C_MP_COMPLETION_SOURCE_SHA256" \
  >"$RUN/logs/mp_completion.out" \
  2>"$RUN/logs/mp_completion.err"

#!/usr/bin/env bash
#SBATCH --job-name=spad-state-reference-verification
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=spad-state-reference-verification-%j.out
#SBATCH --error=spad-state-reference-verification-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
NATIVE="${SPAD_REFERENCE_NATIVE_RUN:?}"
TAU="${SPAD_REFERENCE_TAU_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
test -f "$NATIVE/evaluation/_SUCCESS"
test -f "$TAU/evaluation/_SUCCESS"
RUN="$ROOT/runs/spad_state_reference_verification_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for endpoint in native tau800; do
  if [[ "$endpoint" == native ]]; then
    BASE="$NATIVE"
    PATHS="$NATIVE/native/paths.jsonl"
  else
    BASE="$TAU"
    PATHS="$TAU/tau800/paths.jsonl"
  fi
  "$PY" "$CODE/scripts/audit_programmed_terminal_consistency.py" --labels-jsonl "$BASE/labels/labels.jsonl" \
    --output-dir "$RUN/$endpoint-audit" --gpu-count 2 >"$RUN/$endpoint-audit.out" 2>"$RUN/$endpoint-audit.err"
  "$PY" "$CODE/scripts/apply_programmed_terminal_audit.py" --labels-jsonl "$BASE/labels/labels.jsonl" \
    --audit-jsonl "$RUN/$endpoint-audit/audit.jsonl" --output-dir "$RUN/$endpoint-labels" \
    >"$RUN/$endpoint-apply.out" 2>"$RUN/$endpoint-apply.err"
  "$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$PATHS" \
    --labels-jsonl "$RUN/$endpoint-labels/labels.jsonl" --frozen-config "$CONFIG" \
    --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
    --output-dir "$RUN/$endpoint-evaluation" --expected-requests 256 --endpoint "$endpoint" --cohort-role fixed_development --policy-stage reference \
    >"$RUN/$endpoint-evaluate.out" 2>"$RUN/$endpoint-evaluate.err"
  "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["counts"]; b=json.load(open(sys.argv[2]))["counts"]; keys=("requests","reconstructed","strict_stable","meta_stable","strict_sun","meta_sun"); assert all(a[k]==b[k] for k in keys), "verification-only pass changed the retained energy-threshold metric"' \
    "$BASE/evaluation/EVALUATION_FINAL.json" "$RUN/$endpoint-evaluation/EVALUATION_FINAL.json"
done
printf '%s\n' "$NATIVE" >"$RUN/NATIVE_REFERENCE_RUN"
printf '%s\n' "$TAU" >"$RUN/TAU_REFERENCE_RUN"
touch "$RUN/_SUCCESS"

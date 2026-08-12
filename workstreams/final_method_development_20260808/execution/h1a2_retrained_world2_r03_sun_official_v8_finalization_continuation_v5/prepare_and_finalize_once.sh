#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
STAGED="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_retrained_world2_r03_sun_official_v8_finalization_continuation_v5"
PARENT="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
RUN="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_finalization_continuation_v5"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"
test "$#" -eq 0
test ! -e "$RUN"

test -f "$PARENT/status/combined_official_FAILED"
test "$(cat "$PARENT/status/combined_official_exit_code.txt")" = 1
test -f "$PARENT/status/all_preliminary_cells_SUCCESS"
test -f "$PARENT/status/precompleted_official_cache_SUCCESS"
test -f "$PARENT/precompleted_official_mp_cache/completion_SUCCESS"
test ! -e "$PARENT/inputs"
test ! -e "$PARENT/official_mp_cache"
test ! -e "$PARENT/official_results"
grep -Fq 'args[@]: unbound variable' "$PARENT/logs/h1a2-v8-sunfinal-32049.err"
for index in $(seq 0 8); do
  test "$(cat "$PARENT/status/preliminary_cell_${index}_exit_code.txt")" = 0
  test -f "$PARENT/status/preliminary_cell_${index}_SUCCESS"
  test -f "$PARENT/preliminary/$(printf '%03d' "$index")_"*/_SUCCESS
done

test -f "$STAGED/SOURCE_SHA256.txt"
(cd "$STAGED" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$STAGED/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY CUDA_VISIBLE_DEVICES
"$PYTHON" "$STAGED/self_test.py"
"$PYTHON" "$STAGED/preflight.py" \
  --source-dir "$STAGED" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256"

mkdir -p "$RUN/source" "$RUN/status" "$RUN/logs"
: > "$RUN/status/finalization_continuation.lock"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$RUN/status/finalization_continuation_exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN/status/finalization_continuation_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

cp -a "$STAGED/." "$RUN/source/"
SOURCE="$RUN/source"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
chmod -R a-w "$SOURCE"
cp -al "$PARENT/preliminary" "$RUN/preliminary"
cp -al "$PARENT/precompleted_official_mp_cache" "$RUN/precompleted_official_mp_cache"
for index in $(seq 0 8); do
  cp -p "$PARENT/status/preliminary_cell_${index}_exit_code.txt" "$RUN/status/"
  touch "$RUN/status/preliminary_cell_${index}_SUCCESS"
done
touch "$RUN/status/all_preliminary_cells_SUCCESS"
touch "$RUN/status/precompleted_official_cache_SUCCESS"

"$PYTHON" - "$RUN/status/continuation_lineage.json" "$H1_RETRAINED_SUN_SOURCE_SHA256" <<'PY'
import json, os, sys
path, source_sha = sys.argv[1:]
payload = {
    "schema": "h1a2_v8_official_finalization_continuation_v1",
    "status": "ready",
    "source_manifest_sha256": source_sha,
    "parent_run": "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4",
    "parent_job_id": 32049,
    "parent_state": "FAILED",
    "parent_exit_code": "1:0",
    "parent_failure": "empty_bash_array_expansion_under_nounset_during_official_inputs_freeze",
    "reused_preliminary_cells": 9,
    "reused_official_cache": True,
    "generation_or_refinement_rerun": False,
    "preliminary_evaluation_rerun": False,
    "mp_query_rerun": False,
    "slurm_jobs": 0,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
"$PYTHON" "$SOURCE/preflight.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --output "$RUN/status/preflight_report.json"
touch "$RUN/status/preparation_SUCCESS"

printf 'STAGE official_inputs_audit\n'
"$PYTHON" "$SOURCE/collect_official_inputs.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN" \
  --audit-only
printf 'STAGE official_inputs_freeze\n'
"$PYTHON" "$SOURCE/collect_official_inputs.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN"
printf 'STAGE preliminary_assembly\n'
"$PYTHON" "$SOURCE/assemble_preliminary.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN"
printf 'STAGE adopt_precompleted_official_cache\n'
"$PYTHON" "$SOURCE/adopt_precompleted_cache.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN"
printf 'STAGE official_finalization\n'
"$PYTHON" "$SOURCE/finalize_postonly.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN"

test -f "$RUN/status/finalization_SUCCESS"
test -f "$RUN/official_results/_SUCCESS"
test -f "$RUN/terminal_report.json"
test -f "$RUN/RESULTS_COMPLETE.md"
touch "$RUN/status/finalization_continuation_SUCCESS"
trap - EXIT
printf 'FINALIZATION_CONTINUATION=SUCCESS\nSLURM_JOBS=0\nPRELIMINARY_CELLS_REUSED=9\n'

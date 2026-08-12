#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
SOURCE="$RUN/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"
test "$#" -eq 0
test -f "$RUN/status/preparation_SUCCESS"
test -f "$RUN/status/precompleted_official_cache_SUCCESS"
test -f "$RUN/precompleted_official_mp_cache/completion_SUCCESS"
test ! -e "$RUN/status/combined_submission.lock"
test ! -e "$RUN/status/combined_submission_record.json"
test ! -e "$RUN/status/combined_submission_SUCCESS"
test ! -e "$RUN/official_mp_cache"
test ! -e "$RUN/official_results"
test ! -e "$RUN/terminal_report.json"
test ! -e "$RUN/RESULTS_COMPLETE.md"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
: > "$RUN/status/combined_submission.lock"
on_exit() {
  rc=$?
  if [[ "$rc" -ne 0 ]]; then touch "$RUN/status/combined_submission_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT
job_raw="$(sbatch --parsable \
  --export=ALL,H1_RETRAINED_SUN_SOURCE_SHA256="$H1_RETRAINED_SUN_SOURCE_SHA256" \
  "$SOURCE/combined_official.sbatch")"
JOB="${job_raw%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]]
"$PYTHON" - "$RUN/status/combined_submission_record.json" "$JOB" "$H1_RETRAINED_SUN_SOURCE_SHA256" <<'PY'
import json, os, sys
path, job, source_sha = sys.argv[1:]
payload = {
    "schema": "h1a2_v8_postonly_combined_official_submission_v1",
    "job_id": job,
    "job_name": "h1a2-v8-sunfinal",
    "partition": "gpu",
    "gpus": 4,
    "cpus": 32,
    "memory_gb": 500,
    "cell_waves": [[0, 1, 2, 3], [4, 5, 6, 7], [8]],
    "cell_count": 9,
    "array_jobs": 0,
    "official_slurm_job_count": 1,
    "total_slurm_job_ids_since_v5": 4,
    "source_manifest_sha256": source_sha,
    "evaluated_stage": "post_model494_only",
    "pre_refine_evaluated": False,
    "generation_or_refinement_rerun": False,
    "cuda_reason": "frozen_chgnet_preliminary_requires_cuda",
    "submitted_once": True,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN/status/combined_submission_SUCCESS"
trap - EXIT
printf 'COMBINED_OFFICIAL_JOB=%s\nOFFICIAL_SLURM_JOBS=1\nMAX_A800=4\nMAX_CPU=32\n' "$JOB"

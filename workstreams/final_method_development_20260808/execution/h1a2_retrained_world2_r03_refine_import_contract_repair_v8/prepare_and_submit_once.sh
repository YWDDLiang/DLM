#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
UPSTREAM="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
PREVIOUS="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v6"
UPSTREAM_BODY="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7"
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$RUN_ROOT/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
LEDGER="$PROJECT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl"
BODY_SOURCE="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4/body_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
BODY_CHECKPOINT="$PROJECT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt

test "$#" -eq 0
test ! -e "$RUN_ROOT"
test -f "$STAGING/SOURCE_SHA256.txt"
(cd "$STAGING" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$STAGING/SOURCE_SHA256.txt" | cut -d' ' -f1)"

# V6 was interrupted before submission at the user's request because its
# added preflight was rereading the 6.39 GB body adapter. V8 preserves that
# terminal preparation evidence and performs no large-artifact rehash.
test -f "$PREVIOUS/status/preparation.lock"
test -f "$PREVIOUS/status/preparation_ABORTED_BY_USER_NO_LARGE_REHASH"
test ! -e "$PREVIOUS/status/submission.lock"
test ! -e "$PREVIOUS/status/submission_record.json"
test ! -e "$PREVIOUS/status/submission_SUCCESS"

# V5 is immutable. Its sole Slurm job failed only after all five planners and
# planner assembly completed. Reuse those exact bytes; never sample again.
test -z "$(squeue -h -j 31900 2>/dev/null || true)"
JOB_AUDIT="$(sacct -n -P -j 31900 --format=JobIDRaw,State,ExitCode | awk -F'|' '$1 == "31900" {print $2 "|" $3; exit}')"
test "$JOB_AUDIT" = "FAILED|1:0"
test "$(sha256sum "$UPSTREAM/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 29ca85d57d311f455a4f91d83e6d1c4c896d62b47835958dd37719fdd4d9f162
test -f "$UPSTREAM/status/combined_planner_stage_SUCCESS"
test -f "$UPSTREAM/status/planner_assembly_SUCCESS"
test -f "$UPSTREAM/status/combined_all_FAILED"
test "$(cat "$UPSTREAM/status/combined_all_exit_code.txt")" = 1
for index in 0 1 2 3 4; do
  test -f "$UPSTREAM/status/planner_${index}_SUCCESS"
  test "$(cat "$UPSTREAM/status/planner_${index}_exit_code.txt")" = 0
  test ! -e "$UPSTREAM/status/planner_${index}_FAILED"
done
for index in 0 1 2 3; do
  test -f "$UPSTREAM/status/cell_${index}_FAILED"
  test "$(cat "$UPSTREAM/status/combined_fresh_${index}_child_exit_code.txt")" = 1
  grep -Fq "KeyError: 'adapter_file'" "$UPSTREAM/logs/combined-31900-fresh-${index}.err"
done

test "$(sha256sum "$UPSTREAM/planner_terminal_report.json" | cut -d' ' -f1)" = 5a9676dcc10a4c4938d29aa39bbe509a4635f08f2215d49f756ea165729ca966
test "$(sha256sum "$UPSTREAM/planner_distribution_deep_audit.json" | cut -d' ' -f1)" = 649eddbc90148f74fb11cab865e59a21282750f134d2345522880754f61a9953
test "$(sha256sum "$UPSTREAM/planner_topology_match_audit.json" | cut -d' ' -f1)" = dce0b7cdcb2ad24b3aa7f418e9ede3e3a32689c61e1a7b3cbe34470ac4120b3a
declare -A COHORT_SHA=(
  [retrained_seed52021_world2_b4]=b5a897c029947e1cf88d3abbd2b48efed4cfc1694e4dc73ab9a447ce9f3d178c
  [retrained_seed62023_world2_b4]=3913bae3e07d6a06902dc15e8ff8f484cdf32754b204d5fdaeb86b176c663dd9
  [retrained_seed72031_world2_b4]=daf5b2fa9ec0e00929a91521fac5978e0d8d55008fcf6a2da93ca01224d44b5b
  [retrained_seed82037_world2_b4]=63d764ecb63dd522ffd0fe1844298461fd9525e2169d38c57f2a79e8e8a9c93f
  [retrained_seed17_world2_b4_topology_match]=3a3107489866b551069870cac12a40f02b7c6a8f313845c6df91bc450e529f39
)
for cohort in "${!COHORT_SHA[@]}"; do
  cohort_path="$UPSTREAM/planner/$cohort/frozen/cohort256.jsonl"
  test "$(sha256sum "$cohort_path" | cut -d' ' -f1)" = "${COHORT_SHA[$cohort]}"
done

# V7 completed all four fresh R03 body256 outputs, then failed before any
# model494 refinement because the body runtime PYTHONPATH was still active.
# Reuse only those complete body directories and correct the import root.
test -z "$(squeue -h -j 31965 2>/dev/null || true)"
BODY_JOB_AUDIT="$(sacct -n -P -j 31965 --format=JobIDRaw,State,ExitCode | awk -F'|' '$1 == "31965" {print $2 "|" $3; exit}')"
test "$BODY_JOB_AUDIT" = "FAILED|1:0"
test -f "$UPSTREAM_BODY/status/submission_SUCCESS"
test -f "$UPSTREAM_BODY/status/combined_all_FAILED"
test "$(cat "$UPSTREAM_BODY/status/combined_all_exit_code.txt")" = 1
for index in 0 1 2 3; do
  test -f "$UPSTREAM_BODY/status/cell_${index}_FAILED"
  test "$(cat "$UPSTREAM_BODY/status/cell_${index}_exit_code.txt")" = 1
  test "$(cat "$UPSTREAM_BODY/status/combined_fresh_${index}_child_exit_code.txt")" = 1
  grep -Fq "No module named 'scripts.refine_dlm_with_crysllmgen'" "$UPSTREAM_BODY/logs/combined-31965-fresh-${index}.err"
done

test "$(sha256sum "$LEDGER" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f
test "$(sha256sum "$BODY_SOURCE/run_body_safeaxis1000.py" | cut -d' ' -f1)" = d569d56bc58c8df13a90c7f41847c991b6c6dd671bf1cd6f99fbbe822e50892b
test "$(sha256sum "$BODY_SOURCE/refine1000.py" | cut -d' ' -f1)" = 12c5db21a15b7245d00b4388e2f5204a7dd91b3e10af9831e2a18b347f6cf3a6
test "$(sha256sum "$BODY_SOURCE/finalize_post1000.py" | cut -d' ' -f1)" = 95d664e206166a332824e0b1639365ebde48f00e26a60ed7724b37878f29a6c8
test "$(sha256sum "$R03D/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f
test "$(sha256sum "$R03E/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs" "$RUN_ROOT/inputs"
: > "$RUN_ROOT/status/preparation.lock"
on_exit() {
  rc=$?
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/preparation_or_submission_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT
cp -a "$STAGING" "$SOURCE"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
cp "$LEDGER" "$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
test "$(sha256sum "$RUN_ROOT/inputs/reference_attempt_ledger.jsonl" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/preparation"
export H1_REFERENCE_ATTEMPT_LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
export PYTHONPATH="$SOURCE:$PROJECT"
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY

"$PYTHON" "$SOURCE/self_test.py"
"$PYTHON" - "$UPSTREAM_BODY" "$RUN_ROOT/status/upstream_body_evidence_report.json" <<'PY'
import json, os, sys
from pathlib import Path

upstream = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
cohorts = [
    "retrained_seed52021_world2_b4",
    "retrained_seed62023_world2_b4",
    "retrained_seed72031_world2_b4",
    "retrained_seed82037_world2_b4",
]
expected_succeeded = [246, 242, 250, 245]
required = (
    "batch_partition.json",
    "body_attempts.jsonl",
    "body_tokenizer_identity.json",
    "generation_report.json",
    "input_report.json",
    "proposal_graphs.pt",
)
cells = []
for index, (cohort, expected) in enumerate(zip(cohorts, expected_succeeded)):
    cell = upstream / "cells" / cohort
    body = cell / "body"
    if not body.is_dir() or (cell / "refinement").exists() or (cell / "post_model494").exists():
        raise SystemExit(f"V7 cell stage contract changed: {cohort}")
    sizes = {}
    for name in required:
        path = body / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"missing V7 body evidence: {cohort}/{name}")
        sizes[name] = path.stat().st_size
    report = json.loads((body / "generation_report.json").read_text(encoding="utf-8"))
    attempts = sum(1 for line in (body / "body_attempts.jsonl").open(encoding="utf-8") if line.strip())
    if (
        report.get("status") != "complete"
        or int(report.get("attempts", -1)) != 256
        or int(report.get("succeeded", -1)) != expected
        or int(report.get("repeat", -1)) != index
        or str(report.get("slurm_job_id")) != "31965"
        or report.get("automatic_downstream") is not False
        or report.get("retry_replacement_repair_filter_rerank") is not False
        or attempts != 256
    ):
        raise SystemExit(f"V7 body report contract changed: {cohort}")
    cells.append({
        "cohort": cohort,
        "repeat": index,
        "attempts": attempts,
        "succeeded": expected,
        "files": sizes,
        "refinement_started": False,
    })
payload = {
    "schema": "h1a2_v7_body256_reuse_evidence_v1",
    "status": "pass",
    "upstream_slurm_job_id": "31965",
    "upstream_slurm_terminal": {"state": "FAILED", "exit_code": "1:0"},
    "failure_signature": "ModuleNotFoundError: No module named 'scripts.refine_dlm_with_crysllmgen'",
    "reuse_scope": "four_completed_fresh_r03_body256_outputs_only",
    "body_generation_rerun": False,
    "large_artifact_rehashed": False,
    "cells": cells,
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN_ROOT/status/upstream_body_evidence_SUCCESS"
"$PYTHON" "$SOURCE/audit_body_contract.py" \
  --config "$SOURCE/CONFIG.json" \
  --output "$RUN_ROOT/status/body_runtime_contract_report.json"
touch "$RUN_ROOT/status/body_runtime_contract_SUCCESS"
"$PYTHON" "$SOURCE/audit_refiner_contract.py" \
  --config "$SOURCE/CONFIG.json" \
  --ledger "$RUN_ROOT/inputs/reference_attempt_ledger.jsonl" \
  --checkpoint "$REFINER" \
  --output "$RUN_ROOT/status/refiner_contract_report.json"
touch "$RUN_ROOT/status/refiner_contract_SUCCESS"
export PYTHONPATH="$SOURCE:$R03D/runtime:$R03D"
"$PYTHON" "$SOURCE/run_body_replay.py" \
  --base-script "$BODY_SOURCE/run_body_safeaxis1000.py" \
  --base-script-sha256 d569d56bc58c8df13a90c7f41847c991b6c6dd671bf1cd6f99fbbe822e50892b \
  --import-self-test-only
touch "$RUN_ROOT/status/body_import_isolation_SELF_TEST_SUCCESS"

"$PYTHON" - "$RUN_ROOT/status/upstream_planner_evidence_report.json" <<'PY'
import json, os, sys
path = sys.argv[1]
payload = {
    "schema": "h1a2_v5_planner_reuse_evidence_v1",
    "status": "pass",
    "upstream_slurm_job_id": "31900",
    "upstream_slurm_terminal": {"state": "FAILED", "exit_code": "1:0"},
    "upstream_source_manifest_sha256": "29ca85d57d311f455a4f91d83e6d1c4c896d62b47835958dd37719fdd4d9f162",
    "planner_terminal_sha256": "5a9676dcc10a4c4938d29aa39bbe509a4635f08f2215d49f756ea165729ca966",
    "planner_distribution_sha256": "649eddbc90148f74fb11cab865e59a21282750f134d2345522880754f61a9953",
    "planner_topology_sha256": "dce0b7cdcb2ad24b3aa7f418e9ede3e3a32689c61e1a7b3cbe34470ac4120b3a",
    "planner_sampling_rerun": False,
    "failure_signature": "KeyError: 'adapter_file'",
    "large_artifact_rehashed": False,
    "body_adapter_identity_basis": "registered_sha256_plus_current_path_and_byte_size",
    "refiner_identity_basis": "registered_sha256_plus_current_path_and_byte_size",
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN_ROOT/status/upstream_planner_evidence_SUCCESS"
touch "$RUN_ROOT/status/preparation_SUCCESS"

: > "$RUN_ROOT/status/submission.lock"
job_raw="$(sbatch --parsable \
  --export=ALL,H1_RETRAINED_RECOVERY_SOURCE_SHA256="$SOURCE_SHA" \
  "$SOURCE/combined_generation_all.sbatch")"
JOB="${job_raw%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]]
"$PYTHON" - "$RUN_ROOT/status/submission_record.json" "$JOB" "$SOURCE_SHA" <<'PY'
import json, os, sys
path, job, source_sha = sys.argv[1:]
payload = {
    "schema": "h1a2_retrained_refine_import_continuation_single_job_submission_v8",
    "upstream_planner_job_id": "31900",
    "upstream_body_job_id": "31965",
    "refine_import_continuation_job_id": job,
    "slurm_job_ids_since_v5": ["31900", "31965", job],
    "slurm_job_count_since_v5": 3,
    "remaining_official_slurm_job_budget": 0,
    "planner_sampling_rerun": False,
    "fresh_r03_body_generation_rerun": False,
    "fresh_r03_body_outputs_reused": True,
    "requested_a800_gpus": 4,
    "maximum_visible_a800_gpus": 4,
    "requested_cpus": 32,
    "maximum_concurrent_cpu_threads": 32,
    "post_model494_cells": 9,
    "pre_refine_evaluated": False,
    "large_artifact_rehashed": False,
    "source_manifest_sha256": source_sha,
    "submitted_once": True,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
touch "$RUN_ROOT/status/submission_SUCCESS"
chmod -R a-w "$SOURCE"
trap - EXIT
printf 'REFINE_IMPORT_CONTINUATION=%s\nSLURM_JOB_COUNT_SINCE_V5=3\n' "$JOB"

#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
CANCELLED_V4="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v4"
TRAIN_RUN="$PROJECT/runs/20260812_h1a2_epoch2_exact_retrain_recovery_v1"
STAGING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$RUN_ROOT/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
LEDGER="$PROJECT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl"
BODY_SOURCE="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4/body_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
test "$#" -eq 0
test ! -e "$RUN_ROOT"
test -f "$STAGING/SOURCE_SHA256.txt"
(cd "$STAGING" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$STAGING/SOURCE_SHA256.txt" | cut -d' ' -f1)"

test -f "$CANCELLED_V4/status/submission_SUCCESS"
test -f "$CANCELLED_V4/status/submission_record.json"
test ! -e "$CANCELLED_V4/status/combined_all_SUCCESS"
test -z "$(squeue -h -j 31897 2>/dev/null || true)"
test "$(sha256sum "$TRAIN_RUN/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = e40ba6b88b7df399634b269de8737e2680604d31a46a9b3733566277fc40c782
test -f "$TRAIN_RUN/status/training_SUCCESS"
test "$(cat "$TRAIN_RUN/status/training_exit_code.txt")" = 0
test -f "$TRAIN_RUN/training_terminal_report.json"
mapfile -t TRAIN_VALUES < <("$PYTHON" - "$STAGING/CONFIG.json" "$TRAIN_RUN/training_terminal_report.json" <<'PY'
import json, re, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
terminal = json.load(open(sys.argv[2], encoding="utf-8"))
upstream = config["training_upstream"]
for key in ("adapter_sha256", "adapter_config_sha256"):
    if re.fullmatch(r"[0-9a-f]{64}", str(upstream[key])) is None:
        raise SystemExit(f"unfrozen {key}")
if terminal.get("engineering_status") != "complete":
    raise SystemExit("training is not terminal-complete")
if terminal.get("adapter_sha256") != upstream["adapter_sha256"] or terminal.get("adapter_config_sha256") != upstream["adapter_config_sha256"]:
    raise SystemExit("training terminal/config identity mismatch")
if terminal.get("adapter_path") != upstream["adapter_path"] + "/adapter_model.safetensors":
    raise SystemExit("training terminal adapter path mismatch")
print(upstream["adapter_path"])
print(upstream["adapter_sha256"])
print(upstream["adapter_config_sha256"])
PY
)
CHECKPOINT="${TRAIN_VALUES[0]}"
test "$(sha256sum "$CHECKPOINT/adapter_model.safetensors" | cut -d' ' -f1)" = "${TRAIN_VALUES[1]}"
test "$(sha256sum "$CHECKPOINT/adapter_config.json" | cut -d' ' -f1)" = "${TRAIN_VALUES[2]}"
test "$(sha256sum "$PROJECT/scripts/sample_llama_h1_formula_plans.py" | cut -d' ' -f1)" = d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8
test "$(sha256sum "$PROJECT/crystal_dlm/h1_llm_planner.py" | cut -d' ' -f1)" = d45ccc23fad4284fdeef53d7bbdc5e4044fb6b598092461663473ed8f5a4f8ad
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
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
export H1_REFERENCE_ATTEMPT_LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
export PYTHONPATH="$SOURCE:$PROJECT"
"$PYTHON" "$SOURCE/self_test.py"
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
    "schema": "h1a2_retrained_world2_single_job_submission_v5",
    "combined_generation_job_id": job,
    "generation_slurm_job_count": 1,
    "planner_slurm_array_jobs": 0,
    "planner_cohorts": 5,
    "planner_waves": [[0, 1], [2, 3], [4]],
    "requested_a800_gpus": 4,
    "maximum_visible_a800_gpus": 4,
    "requested_cpus": 32,
    "maximum_concurrent_cpu_threads": 32,
    "post_model494_cells": 9,
    "pre_refine_evaluated": False,
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
printf 'COMBINED_GENERATION_ALL=%s\nGENERATION_SLURM_JOB_COUNT=1\n' "$JOB"

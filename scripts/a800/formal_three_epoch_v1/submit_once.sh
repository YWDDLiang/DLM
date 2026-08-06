#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BATCH=configs/experiments/wyckoff_codiffusion/formal_three_epoch_dag_v1.json
BATCH_SHA=d65d63ebad4b9db2f518df690f7a94fff054d35b51069e978bf245ebd2f27512
RECORD="$RUN/notes/formal_three_epoch_dag_v1_submission.json"
JOB_DIR=scripts/a800/formal_three_epoch_v1

: "${EXECUTION_PATCH_SHA256:?caller must export the installed patch-manifest SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid execution patch SHA256" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test "$(sha256sum "$BATCH" | awk '{print $1}')" = "$BATCH_SHA"
test ! -e "$RECORD"
for output in \
  "$RUN/outputs/ddp_flash_followup_v1/cache4_mb8_acc4_gc_off_flash2" \
  "$RUN/notes/formal_execution_selection_v1.json" \
  "$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1" \
  "$RUN/outputs/train_wq_refiner_seed11_formal_supersession26679_v1" \
  "$RUN/outputs/epoch_checkpoint_panel_v1" \
  "$RUN/notes/epoch_checkpoint_evidence_v1.json" \
  "$RUN/outputs/epoch_checkpoint_selection_lock_v1.json"; do
  test ! -e "$output"
done

python - "$BATCH" <<'PY'
import hashlib, json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "crysllmgen_formal_three_epoch_dependency_dag_v1":
    raise SystemExit("formal DAG schema changed")
if payload.get("status") != "active_user_authorized_ready_to_submit_once":
    raise SystemExit("formal DAG is not active")
for job in payload["jobs"]:
    path = Path(job["script"])
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != job["script_sha256"]:
        raise SystemExit(f"formal DAG script changed: {path}")
if payload["submission_accounting"]["slurm_submit_slots_including_array_elements"] != 7:
    raise SystemExit("formal DAG submit-slot denominator changed")
print("formal_dag_manifest_audit=PASS")
PY

mapfile -t existing_rows < <(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)
if [ "${#existing_rows[@]}" -gt 1 ]; then
  echo "preflight refuses 7 new submit slots with ${#existing_rows[@]} existing jobs" >&2
  printf '%s\n' "${existing_rows[@]}" >&2
  exit 3
fi
gpu_job_ids=()
for row in "${existing_rows[@]}"; do
  IFS='|' read -r job_id partition gres name <<<"$row"
  if [[ "$gres" == *gpu* ]]; then
    gpu_job_ids+=("${job_id%%_*}")
  fi
done
if [ "${#gpu_job_ids[@]}" -gt 1 ]; then
  echo "preflight found more than one existing GPU job" >&2
  exit 3
fi

existing_dependency=()
if [ "${#gpu_job_ids[@]}" -eq 1 ]; then
  existing_dependency=(--dependency="afterany:${gpu_job_ids[0]}")
fi

flash_job=""
train_job=""
refiner_job=""
eval_job=""
selection_job=""

write_record() {
  local status="$1"
  local failed_stage="$2"
  local failure_message="$3"
  python - "$RECORD" "$status" "$failed_stage" "$failure_message" \
    "$EXECUTION_PATCH_SHA256" "$BATCH_SHA" "$flash_job" "$train_job" \
    "$refiner_job" "$eval_job" "$selection_job" \
    "$(printf '%s\n' "${existing_rows[@]}")" <<'PY'
import json, os, sys
from pathlib import Path

(
    path, status, failed_stage, failure_message, patch_sha, batch_sha,
    flash_job, train_job, refiner_job, eval_job, selection_job, existing,
) = sys.argv[1:]
payload = {
    "schema": "crysllmgen_formal_three_epoch_dag_submission_v1",
    "status": status,
    "failed_stage": failed_stage or None,
    "failure_message": failure_message or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_patch_sha256": patch_sha,
    "batch_manifest": {
        "path": "configs/experiments/wyckoff_codiffusion/formal_three_epoch_dag_v1.json",
        "sha256": batch_sha,
    },
    "jobs": {
        "flash_profile": flash_job or None,
        "formal_train": train_job or None,
        "formal_refiner_supersession": refiner_job or None,
        "epoch_evaluation_array": eval_job or None,
        "epoch_selection": selection_job or None,
    },
    "preexisting_jobs": [line for line in existing.splitlines() if line],
    "maximum_concurrent_a800": 2,
    "new_submit_slots_including_array_elements": 7,
    "retry_or_replacement_policy": "no_resubmission; only refiner supersedes failed job 26679",
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

submit_or_record_failure() {
  local variable_name="$1"
  local stage="$2"
  shift 2
  local output rc
  set +e
  output=$(sbatch --parsable "$@" 2>&1)
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] || [[ ! "$output" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    write_record partial_submission_failed "$stage" "rc=$rc output=$output"
    echo "submission failed at $stage: $output" >&2
    exit 4
  fi
  printf -v "$variable_name" '%s' "${output%%_*}"
}

submit_or_record_failure flash_job flash_profile \
  "${existing_dependency[@]}" \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256" \
  "$JOB_DIR/flash_profile.sbatch"

submit_or_record_failure train_job formal_train \
  --dependency="afterany:$flash_job" \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256,FLASH_PROFILE_JOB_ID=$flash_job" \
  "$JOB_DIR/formal_train.sbatch"

submit_or_record_failure refiner_job formal_refiner_supersession \
  --dependency="afterok:$train_job" \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256" \
  "$JOB_DIR/refiner.sbatch"

submit_or_record_failure eval_job epoch_evaluation_array \
  --dependency="afterok:$refiner_job" \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256" \
  "$JOB_DIR/epoch_eval.sbatch"

submit_or_record_failure selection_job epoch_selection \
  --dependency="afterok:$eval_job" \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256" \
  "$JOB_DIR/select_epoch.sbatch"

write_record complete "" ""
cat "$RECORD"

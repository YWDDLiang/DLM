#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BATCH=configs/experiments/wyckoff_codiffusion/refiner_supersession26955_v2_dag.json
BATCH_SHA=500a5d035110b905f024364c1cd4b07340ab8eb38af03da478a449318324491e
RECORD="$RUN/notes/refiner_supersession26955_v2_submission.json"
JOB_DIR=scripts/a800/refiner_supersession26955_v2
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
REFINER_OUTPUT="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26955_v2"
REFINER_CHECKPOINT="$REFINER_OUTPUT/model_ema_final.pt"
PANEL="$RUN/outputs/epoch_checkpoint_panel_sup26955_v2"
EVIDENCE="$RUN/notes/epoch_checkpoint_evidence_sup26955_v2.json"
SELECTION="$RUN/outputs/epoch_checkpoint_selection_lock_sup26955_v2.json"

: "${EXECUTION_PATCH_SHA256:?caller must export the installed patch-manifest SHA256}"
if [[ ! "$EXECUTION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid execution patch SHA256" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EXECUTION_PATCH_SHA256}.json"
test "$(sha256sum "$BATCH" | awk '{print $1}')" = "$BATCH_SHA"
test "$(sha256sum "$TRAIN/training_report.json" | awk '{print $1}')" = 04d6b6f78668266508102ce018c3a4afc4077379d9f7d12f0eeffa661f4ba430
test -f "$TRAIN/epoch_01/adapter_final/adapter_model.safetensors"
test -f "$TRAIN/epoch_02/adapter_final/adapter_model.safetensors"
test -f "$TRAIN/epoch_03/adapter_final/adapter_model.safetensors"
test ! -e "$RECORD"
for output in \
  "$RUN/outputs/refiner_prefetch7_stability_smoke_v2" \
  "$REFINER_OUTPUT" \
  "$PANEL" \
  "$EVIDENCE" \
  "$SELECTION"; do
  test ! -e "$output"
done

python - "$BATCH" <<'PY'
import hashlib, json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "crysllmgen_refiner_supersession_dependency_dag_v1":
    raise SystemExit("refiner supersession DAG schema changed")
if payload.get("status") != "active_user_authorized_ready_to_submit_once":
    raise SystemExit("refiner supersession DAG is not active")
for job in payload["jobs"]:
    path = Path(job["script"])
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != job["script_sha256"]:
        raise SystemExit(f"refiner supersession DAG script changed: {path}")
accounting = payload["submission_accounting"]
if (
    accounting["slurm_submit_slots_including_array_elements"] != 6
    or accounting["maximum_concurrent_a800"] != 2
    or payload["prefetch_selection"]["workers"] != 7
    or payload["prefetch_selection"]["depth"] != 14
):
    raise SystemExit("refiner supersession DAG accounting changed")
print("refiner_supersession_dag_manifest_audit=PASS")
PY

mapfile -t existing_rows < <(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)
if [ "${#existing_rows[@]}" -gt 1 ]; then
  echo "preflight refuses 6 new submit slots with ${#existing_rows[@]} existing jobs" >&2
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

smoke_job=""
refiner_job=""
eval_job=""
selection_job=""

write_record() {
  local status="$1"
  local failed_stage="$2"
  local failure_message="$3"
  python - "$RECORD" "$status" "$failed_stage" "$failure_message" \
    "$EXECUTION_PATCH_SHA256" "$BATCH_SHA" "$smoke_job" "$refiner_job" \
    "$eval_job" "$selection_job" "$(printf '%s\n' "${existing_rows[@]}")" <<'PY'
import json, os, sys
from pathlib import Path

(
    path, status, failed_stage, failure_message, patch_sha, batch_sha,
    smoke_job, refiner_job, eval_job, selection_job, existing,
) = sys.argv[1:]
payload = {
    "schema": "crysllmgen_refiner_supersession_dag_submission_v1",
    "status": status,
    "failed_stage": failed_stage or None,
    "failure_message": failure_message or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_patch_sha256": patch_sha,
    "batch_manifest": {
        "path": "configs/experiments/wyckoff_codiffusion/refiner_supersession26955_v2_dag.json",
        "sha256": batch_sha,
    },
    "jobs": {
        "prefetch_stability_smoke": smoke_job or None,
        "formal_refiner_supersession26955": refiner_job or None,
        "epoch_evaluation_array": eval_job or None,
        "epoch_selection": selection_job or None,
    },
    "preexisting_jobs": [line for line in existing.splitlines() if line],
    "maximum_concurrent_a800": 2,
    "new_submit_slots_including_array_elements": 6,
    "retry_or_replacement_policy": "no resubmission; exactly one refiner supersession of failed job 26955",
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

submit_or_record_failure smoke_job prefetch_stability_smoke \
  "${existing_dependency[@]}" \
  --job-name=wq-ref-smoke-p7-v2 \
  --time=00:30:00 \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256,WQ_REFINER_MODE=smoke" \
  "$JOB_DIR/refiner_candidate.sbatch"

submit_or_record_failure refiner_job formal_refiner_supersession26955 \
  --dependency="afterok:$smoke_job" \
  --job-name=wq-ref-sup26955-v2 \
  --time=08:00:00 \
  --export="ALL,EXECUTION_PATCH_SHA256=$EXECUTION_PATCH_SHA256,WQ_REFINER_MODE=main" \
  "$JOB_DIR/refiner_candidate.sbatch"

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

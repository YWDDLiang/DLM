#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
BATCH=configs/experiments/wyckoff_codiffusion/epoch_eval_selection_supersession27104_v2_dag.json
BATCH_SHA=28ebedcb62e3f60209fb20c02dff9196112dd958ee806142ae90d0d9f9d47673
AUTHORIZATION=runs/remote_audit/20260724_user_authorization_epoch_eval_selection_supersession27104_v2_bash42.json
FAILURE_AUDIT=runs/remote_audit/20260723_epoch_eval_selection_supersession27104_v1_presbatch_failure_v1.json
RECORD="$RUN/notes/epoch_eval_selection_supersession27104_v2_submission.json"
CLAIM="$RECORD.claim"
PRIOR_RECORD="$RUN/notes/epoch_eval_selection_supersession27104_v1_submission.json"
PRIOR_CLAIM="$PRIOR_RECORD.claim"
JOB_DIR=scripts/a800/epoch_eval_selection_supersession27104_v2
TRAIN="$RUN/outputs/train_wq_lora_seed11_mixed_three_epoch_formal_v1"
REFINER="$RUN/outputs/train_wq_refiner_seed11_formal_supersession26955_v2"
PANEL="$RUN/outputs/epoch_checkpoint_panel_sup27104_v2"
EVIDENCE="$RUN/notes/epoch_checkpoint_evidence_sup27104_v2.json"
SELECTION="$RUN/outputs/epoch_checkpoint_selection_lock_sup27104_v2.json"
PRIOR_PANEL="$RUN/outputs/epoch_checkpoint_panel_sup27104_v1"
PRIOR_EVIDENCE="$RUN/notes/epoch_checkpoint_evidence_sup27104_v1.json"
PRIOR_SELECTION="$RUN/outputs/epoch_checkpoint_selection_lock_sup27104_v1.json"
PRIOR_EVALUATION_PATCH_SHA256=5dd8ec04f3834249af8ef1d18ee43ca114d1038a9025024320ab9b93e80da958
ADAPTER_TRAINING_PATCH_SHA256=9577e00d71f38cecd585d6fd228a22794fc0f474014ba3b847e2417fb6121491
REFINER_TRAINING_PATCH_SHA256=a868de4ac99d833628eca08304b6ca6066be58f138b5f03d1dd0e508a61beadc

: "${EVALUATION_PATCH_SHA256:?caller must export the installed evaluation patch-manifest SHA256}"
if [[ ! "$EVALUATION_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$EVALUATION_PATCH_SHA256" = "$PRIOR_EVALUATION_PATCH_SHA256" ] \
  || [ "$EVALUATION_PATCH_SHA256" = "$ADAPTER_TRAINING_PATCH_SHA256" ] \
  || [ "$EVALUATION_PATCH_SHA256" = "$REFINER_TRAINING_PATCH_SHA256" ]; then
  echo "invalid or non-superseding evaluation patch SHA256" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${EVALUATION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${PRIOR_EVALUATION_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${ADAPTER_TRAINING_PATCH_SHA256}.json"
test -f ".artifacts/source_sync/authorized_patch_${REFINER_TRAINING_PATCH_SHA256}.json"
test "$(sha256sum "$BATCH" | awk '{print $1}')" = "$BATCH_SHA"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = 81b4c80cca71e15e75a6b4e7658485d918717aba1956be4c5761ef5d4a25a2f3
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = b4a02875ce58af1b8e33d774ee6005013cbd4e5a1e95ebff2ca3268ae9781d39
test "$(sha256sum "$TRAIN/training_report.json" | awk '{print $1}')" = 04d6b6f78668266508102ce018c3a4afc4077379d9f7d12f0eeffa661f4ba430
test "$(sha256sum "$REFINER/training_report.json" | awk '{print $1}')" = 87e9b64a7154b8b451b118f18f80f4bcb575550cbef2b67fc0c9d507a9a70316
test "$(sha256sum "$REFINER/model_ema_final.pt" | awk '{print $1}')" = 8abb6ba966e0a7153fecf7141c97e659977692bc74ee4573317cbb26bddc4b70
test ! -e "$RECORD"
test ! -e "$CLAIM"
test ! -e "$PRIOR_RECORD"
test ! -e "$PRIOR_CLAIM"
test ! -e "$PRIOR_PANEL"
test ! -e "$PRIOR_EVIDENCE"
test ! -e "$PRIOR_SELECTION"
test ! -e "$PANEL"
test ! -e "$EVIDENCE"
test ! -e "$SELECTION"

python - "$BATCH" "$TRAIN" "$REFINER/training_report.json" \
  "$ADAPTER_TRAINING_PATCH_SHA256" "$REFINER_TRAINING_PATCH_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

batch_path, training_root, refiner_report_path = map(Path, sys.argv[1:4])
adapter_patch, refiner_patch = sys.argv[4:]
batch = json.loads(batch_path.read_text(encoding="utf-8"))
if (
    batch.get("schema")
    != "crysllmgen_epoch_eval_selection_supersession_dag_v2"
    or batch.get("status") != "active_user_authorized_ready_to_submit_once"
    or batch["failure_being_superseded"]["entrypoint_identity"]
    != "epoch_eval_selection_supersession27104_v1"
    or batch["failure_being_superseded"]["epoch_evaluation_array_job_id"]
    != "27104"
    or batch["failure_being_superseded"]["prior_sbatch_invoked"] is not False
    or batch["failure_being_superseded"]["scientific_attempts_started"] != 0
    or batch["compatibility_revision"]["target_shell"]
    != "GNU Bash 4.2.46(2)-release"
    or batch["compatibility_revision"]["exclusive_claim_before_any_sbatch"]
    is not True
    or batch["submission_accounting"]["slurm_submit_slots_including_array_elements"]
    != 4
    or batch["submission_accounting"]["maximum_concurrent_a800"] != 2
):
    raise SystemExit("evaluation supersession DAG contract changed")
for job in batch["jobs"]:
    path = Path(job["script"])
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != job["script_sha256"]:
        raise SystemExit(f"evaluation supersession script changed: {path}")
expected_reports = {
    1: "f68a60de6909f5268c741816b4a754f704c3f22652be0676692174d19f616ad7",
    2: "23a9fe46b9b4ca75198c4c20b9b592a0f85e6e86b17b4b422cc287f678b49363",
    3: "1d4bb3275dd5f1a20b9743161b59c633faad6f008ebbf7881ad07ddd2c6dfaf9",
}
for epoch, expected_sha in expected_reports.items():
    path = training_root / f"epoch_{epoch:02d}" / "training_report.json"
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        observed_sha != expected_sha
        or report.get("logical_epoch") != epoch
        or report.get("global_step") != 848 * epoch
        or report.get("execution_patch_sha256") != adapter_patch
        or report.get("slurm_job_id") != "26954"
    ):
        raise SystemExit(f"epoch {epoch} immutable adapter identity changed")
refiner = json.loads(refiner_report_path.read_text(encoding="utf-8"))
if (
    refiner.get("ok") is not True
    or refiner.get("paper_eligible") is not True
    or refiner.get("updates") != 100000
    or refiner.get("execution_patch_sha256") != refiner_patch
    or refiner.get("replacement_of_job_id") != "26955"
):
    raise SystemExit("immutable refiner identity changed")
print("evaluation_supersession_dag_identity_audit=PASS")
PY

# BASH42_QUEUE_PREFLIGHT_BEGIN
existing_rows="$(
  squeue -r -u "$USER" -h -t PENDING,RUNNING -o '%i|%P|%b|%j' | sort -u
)"
gpu_rows="$(
  printf '%s\n' "$existing_rows" |
    awk -F '|' 'NF >= 3 && $3 ~ /gpu/ { print }'
)"
if [ -n "$gpu_rows" ]; then
  echo "preflight waits for zero preexisting user GPU jobs" >&2
  printf '%s\n' "$gpu_rows" >&2
  exit 3
fi
# BASH42_QUEUE_PREFLIGHT_END

eval_job=""
selection_job=""
eval_command=""
selection_command=""

write_record() {
  local status="$1"
  local failed_stage="$2"
  local failure_message="$3"
  python - "$RECORD" "$status" "$failed_stage" "$failure_message" \
    "$EVALUATION_PATCH_SHA256" "$ADAPTER_TRAINING_PATCH_SHA256" \
    "$REFINER_TRAINING_PATCH_SHA256" "$PRIOR_EVALUATION_PATCH_SHA256" \
    "$BATCH_SHA" "$eval_job" "$selection_job" "$eval_command" \
    "$selection_command" "$existing_rows" <<'PY'
import json, os, sys
from pathlib import Path

(
    path, status, failed_stage, failure_message, evaluation_patch,
    adapter_patch, refiner_patch, prior_evaluation_patch, batch_sha,
    eval_job, selection_job, eval_command, selection_command, existing,
) = sys.argv[1:]
payload = {
    "schema": "crysllmgen_epoch_eval_selection_supersession_submission_v2",
    "status": status,
    "failed_stage": failed_stage or None,
    "failure_message": failure_message or None,
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_identities": {
        "adapter_training_execution_patch_sha256": adapter_patch,
        "refiner_training_execution_patch_sha256": refiner_patch,
        "evaluation_execution_patch_sha256": evaluation_patch,
        "prior_evaluation_execution_patch_sha256": prior_evaluation_patch,
    },
    "batch_manifest": {
        "path": (
            "configs/experiments/wyckoff_codiffusion/"
            "epoch_eval_selection_supersession27104_v2_dag.json"
        ),
        "sha256": batch_sha,
    },
    "supersedes": {
        "entrypoint_identity": "epoch_eval_selection_supersession27104_v1",
        "epoch_evaluation_array_job_id": "27104",
        "selection_job_id": "27105",
        "prior_claim_created": False,
        "prior_sbatch_invoked": False,
        "scientific_attempts_started": 0,
    },
    "jobs": {
        "epoch_evaluation_array": {
            "job_id": eval_job or None,
            "sbatch_command": eval_command or None,
        },
        "epoch_selection": {
            "job_id": selection_job or None,
            "sbatch_command": selection_command or None,
        },
    },
    "preexisting_jobs": [line for line in existing.splitlines() if line],
    "maximum_concurrent_a800": 2,
    "new_submit_slots_including_array_elements": 4,
    "retry_or_replacement_used": True,
    "attempt_retry_or_replacement_used": False,
    "retry_or_replacement_policy": (
        "one explicitly authorized Bash-4.2 compatibility supersession "
        "of a pre-claim pre-sbatch entry; no task resubmission or "
        "attempt replacement"
    ),
    "submitted_by_slurm_user": os.environ.get("USER"),
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

submit_or_record_failure() {
  local variable_name="$1"
  local command_variable_name="$2"
  local stage="$3"
  shift 3
  local command_display output rc
  printf -v command_display '%q ' sbatch --parsable "$@"
  command_display=${command_display% }
  printf -v "$command_variable_name" '%s' "$command_display"
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

mkdir -p "$RUN/notes"
python - "$CLAIM" "$EVALUATION_PATCH_SHA256" \
  "$ADAPTER_TRAINING_PATCH_SHA256" "$REFINER_TRAINING_PATCH_SHA256" \
  "$BATCH_SHA" <<'PY'
import json, os, sys
from pathlib import Path

path, evaluation_patch, adapter_patch, refiner_patch, batch_sha = sys.argv[1:]
payload = {
    "schema": "crysllmgen_epoch_eval_selection_submission_claim_v2",
    "status": "claimed_before_any_sbatch",
    "run_id": "20260720_0401-crysllmgen-wq-final-v3",
    "execution_identities": {
        "adapter_training_execution_patch_sha256": adapter_patch,
        "refiner_training_execution_patch_sha256": refiner_patch,
        "evaluation_execution_patch_sha256": evaluation_patch,
    },
    "batch_manifest_sha256": batch_sha,
    "submitted_by_slurm_user": os.environ.get("USER"),
    "claim_policy": (
        "exclusive-create before any sbatch; retained permanently to prevent "
        "duplicate submission"
    ),
}
with Path(path).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

identity_export="ALL,EVALUATION_PATCH_SHA256=$EVALUATION_PATCH_SHA256,ADAPTER_TRAINING_PATCH_SHA256=$ADAPTER_TRAINING_PATCH_SHA256,REFINER_TRAINING_PATCH_SHA256=$REFINER_TRAINING_PATCH_SHA256"

submit_or_record_failure eval_job eval_command epoch_evaluation_array \
  --export="$identity_export" \
  "$JOB_DIR/epoch_eval.sbatch"

submit_or_record_failure selection_job selection_command epoch_selection \
  --dependency="afterok:$eval_job" \
  --export="$identity_export" \
  "$JOB_DIR/select_epoch.sbatch"

write_record complete "" ""
cat "$RECORD"

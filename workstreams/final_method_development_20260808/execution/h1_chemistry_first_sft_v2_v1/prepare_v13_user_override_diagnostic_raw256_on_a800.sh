#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_V10="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
PARENT_V12="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_slurm_list_serialization_repair_v12"
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SELF_SHA256="${1:?expected preparation-script SHA256}"
SELF="$(realpath "${BASH_SOURCE[0]}")"

EXECUTION_REL=workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1
SOURCE_ROOT="${PARENT_V10}/source"
EXECUTION_DIR="${SOURCE_ROOT}/${EXECUTION_REL}"
EXPECTED_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_SOURCE_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_TRAINING_TERMINAL_SHA256=30575d66321375cf0af3a7aef44f80283e22bf887bfbbc501cb434a75090ed93
EXPECTED_RAW64_TERMINAL_SHA256=5342a4f8e0f5695bfff8680406569d55916cdb66ce8a1d7aabd9f6c2d06a9f0c
EXPECTED_RAW64_STAGE_SUMMARY_SHA256=9b922c112ce707712706d27842a5450efd4bd870594bcb5c241a76e21a8d54da
EXPECTED_LEDGER256_SHA256=d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669

test -x "${PYTHON}"
test -f "${SELF}"
test "$(sha256sum "${SELF}" | cut -d' ' -f1)" = "${EXPECTED_SELF_SHA256}"
test ! -e "${RUN_ROOT}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${PARENT_V10}/source_archive.tar.gz" | cut -d' ' -f1)" = \
  "${EXPECTED_SOURCE_ARCHIVE_SHA256}"
test "$(sha256sum "${PARENT_V10}/training/sft_v2/terminal_report.json" | cut -d' ' -f1)" = \
  "${EXPECTED_TRAINING_TERMINAL_SHA256}"
test "$(sha256sum "${PARENT_V12}/planner64/terminal/sft_v2_terminal_report.json" | cut -d' ' -f1)" = \
  "${EXPECTED_RAW64_TERMINAL_SHA256}"
test "$(sha256sum "${PARENT_V12}/planner64/terminal/stage_summary.json" | cut -d' ' -f1)" = \
  "${EXPECTED_RAW64_STAGE_SUMMARY_SHA256}"
test "$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)" = \
  "${EXPECTED_LEDGER256_SHA256}"
test -f "${PARENT_V10}/status/train_sft_v2_SUCCESS"

mkdir -p "${RUN_ROOT}/launchers" "${RUN_ROOT}/logs" "${RUN_ROOT}/preflight" \
  "${RUN_ROOT}/status"
ln -s "${SOURCE_ROOT}" "${RUN_ROOT}/source"
ln -s "${PARENT_V10}/source_archive.tar.gz" "${RUN_ROOT}/source_archive.tar.gz"
ln -s "${PARENT_V10}/training" "${RUN_ROOT}/training"
cp "${EXECUTION_DIR}/LEDGER256.json" "${RUN_ROOT}/LEDGER256.json"
cp "${SELF}" "${RUN_ROOT}/prepare_v13_user_override_diagnostic_raw256_on_a800.sh"

cat > "${RUN_ROOT}/AUTHORIZATION.json" <<'JSON'
{
  "schema": "h1_chemistry_first_sft_v2_user_override_diagnostic_raw256_v13",
  "status": "user_authorized",
  "authorized_arms": ["p0", "sft_v2"],
  "reason": "formal same-ledger raw64 legacy composition validity improved from 34/64 to 52/64",
  "scope": {
    "planner_raw256": true,
    "local_exact_smact4_secondary_ledger": true,
    "a800_smact31_assembly": true,
    "protected_b0_safe_axis_model494_direct_sun256": true
  },
  "decision_firewall": {
    "diagnostic_only": true,
    "raw64_formal_promotion_rewritten": false,
    "sft_v2_c_downstream": false,
    "automatic_rl": false,
    "automatic_final_promotion": false
  },
  "sampling_contract": {
    "denominator": 256,
    "base_seed": 26081256,
    "seed_mode": "stateless_ordinal_v1",
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 96,
    "batch_size": 1,
    "retry_replacement_repair_filter_rerank": false
  }
}
JSON

cat > "${RUN_ROOT}/launchers/planner256_v13.sbatch" <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=h1-cf-p256-v13
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13/logs/%A_%a_planner256.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13/logs/%A_%a_planner256.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PARENT_V10="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
PARENT_V12="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_slurm_list_serialization_repair_v12"
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13"
SOURCE_ROOT="${PARENT_V10}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
STAGE=256
BASE_SEED=26081256

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${EXPECTED_LEDGER_SHA256:?missing planner256 ledger identity}"
: "${EXPECTED_AUTHORIZATION_SHA256:?missing authorization identity}"
: "${EXPECTED_TRAINING_TERMINAL_SHA256:?missing training identity}"
: "${EXPECTED_RAW64_TERMINAL_SHA256:?missing raw64 terminal identity}"
: "${EXPECTED_RAW64_STAGE_SUMMARY_SHA256:?missing raw64 stage identity}"
: "${LEGACY_PYTHON:?missing generation runtime}"
: "${SLURM_ARRAY_TASK_ID:?missing arm identity}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=p0; PROMPT_STYLE=h1_rich_plan_v1; CHECKPOINT="${P0_ADAPTER}" ;;
  1) ARM=sft_v2; PROMPT_STYLE=h1_rich_nocharge_plan_v1 ;;
  *) exit 3 ;;
esac

test "${SLURM_JOB_PARTITION:-}" = gpu
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = \
  "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${RUN_ROOT}/LEDGER256.json" | cut -d' ' -f1)" = \
  "${EXPECTED_LEDGER_SHA256}"
test "$(sha256sum "${RUN_ROOT}/AUTHORIZATION.json" | cut -d' ' -f1)" = \
  "${EXPECTED_AUTHORIZATION_SHA256}"
test "$(sha256sum "${PARENT_V10}/training/sft_v2/terminal_report.json" | cut -d' ' -f1)" = \
  "${EXPECTED_TRAINING_TERMINAL_SHA256}"
test "$(sha256sum "${PARENT_V12}/planner64/terminal/sft_v2_terminal_report.json" | cut -d' ' -f1)" = \
  "${EXPECTED_RAW64_TERMINAL_SHA256}"
test "$(sha256sum "${PARENT_V12}/planner64/terminal/stage_summary.json" | cut -d' ' -f1)" = \
  "${EXPECTED_RAW64_STAGE_SUMMARY_SHA256}"
test ! -e "${RUN_ROOT}/planner256/${ARM}"

if test "${ARM}" = sft_v2; then
  test -f "${PARENT_V10}/status/train_sft_v2_SUCCESS"
  CHECKPOINT="$("${LEGACY_PYTHON}" "${EXECUTION_DIR}/resolve_fixed_adapter.py" \
    --training-dir "${PARENT_V10}/training/sft_v2" --candidate sft_v2)"
fi

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/planner256" "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/planner256_${ARM}_exit_code.txt"
  touch "${RUN_ROOT}/status/planner256_${ARM}_FAILED"
  exit "${rc}"
}
trap on_error ERR

cd "${SOURCE_ROOT}"
"${LEGACY_PYTHON}" scripts/sample_llama_h1_formula_plans.py \
  --model-path "${MODEL_PATH}" --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner256/${ARM}" \
  --num-samples "${STAGE}" --batch-size 1 --max-new-tokens 96 \
  --temperature 0.9 --top-p 0.95 --top-k 50 --max-atoms 20 \
  --prompt-style "${PROMPT_STYLE}" --no-include-sample-id \
  --seed "${BASE_SEED}" --seed-mode stateless_ordinal_v1 \
  --formula-constraint-mode off --formula-missing-state-policy allow_non_applicable
test "$(wc -l < "${RUN_ROOT}/planner256/${ARM}/raw_generations.jsonl")" -eq 256
printf '0\n' > "${RUN_ROOT}/status/planner256_${ARM}_exit_code.txt"
touch "${RUN_ROOT}/status/planner256_${ARM}_SUCCESS"
echo "H1_CHEMISTRY_FIRST_DIAGNOSTIC_PLANNER256_ARM_PASS arm=${ARM}"
SBATCH

cat > "${RUN_ROOT}/launchers/submit_v13_once.sh" <<'SUBMIT'
#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_user_override_diagnostic_raw256_v13"
PARENT_V10="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_gpu_partition_cancel_array_parser_repair_v10"
PARENT_V12="${PROJECT_ROOT}/runs/20260809_h1_chemistry_first_sft_v2_v10_slurm_list_serialization_repair_v12"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPECTED_SOURCE_INVENTORY_SHA256=4d8e7bdeeb50aaa175e6b7620ef7ba84c882de5f9f5333db77511d2c9c231c60
EXPECTED_SOURCE_ARCHIVE_SHA256=cb1f33dab60d65448dfa92c2f1ef7b13f41307481cb4cf77989e11e23faee87b
EXPECTED_TRAINING_TERMINAL_SHA256=30575d66321375cf0af3a7aef44f80283e22bf887bfbbc501cb434a75090ed93
EXPECTED_RAW64_TERMINAL_SHA256=5342a4f8e0f5695bfff8680406569d55916cdb66ce8a1d7aabd9f6c2d06a9f0c
EXPECTED_RAW64_STAGE_SUMMARY_SHA256=9b922c112ce707712706d27842a5450efd4bd870594bcb5c241a76e21a8d54da
EXPECTED_LEDGER_SHA256=d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669

test -f "${RUN_ROOT}/status/preparation_SUCCESS"
sha256sum -c "${RUN_ROOT}/V13_SHA256.txt"
test ! -e "${RUN_ROOT}/submission_record.json"
test ! -e "${RUN_ROOT}/planner256"
mkdir "${RUN_ROOT}/.submit_v13_lock"

AUTHORIZATION_SHA256="$(sha256sum "${RUN_ROOT}/AUTHORIZATION.json" | cut -d' ' -f1)"
"${PYTHON}" - "${RUN_ROOT}/AUTHORIZATION.json" "${RUN_ROOT}/LEDGER256.json" <<'PY'
import json, sys
auth = json.load(open(sys.argv[1], encoding="utf-8"))
ledger = json.load(open(sys.argv[2], encoding="utf-8"))
assert auth["status"] == "user_authorized"
assert auth["authorized_arms"] == ["p0", "sft_v2"]
assert auth["decision_firewall"]["diagnostic_only"] is True
assert auth["decision_firewall"]["automatic_rl"] is False
assert auth["sampling_contract"]["denominator"] == 256
assert ledger["stage"] == "planner256"
assert ledger["denominator"] == 256
assert ledger["base_seed"] == 26081256
assert len(ledger["rows"]) == 256
assert [row["ordinal"] for row in ledger["rows"]] == list(range(256))
PY

partition_snapshot="$(sinfo -h -p gpu -o '%P|%a|%l|%G' | sed 's/[*]//g')"
printf '%s\n' "${partition_snapshot}" | awk -F'|' \
  '$1 == "gpu" && $2 == "up" {ok=1} END {exit ok ? 0 : 1}'
printf '%s\n' "${partition_snapshot}" > "${RUN_ROOT}/status/sinfo_before_v13.txt"
squeue -h -u "${USER}" -o '%i|%j|%T|%M|%l|%P|%b|%R' \
  > "${RUN_ROOT}/status/squeue_before_v13.txt"

common_export="ALL,EXPECTED_SOURCE_INVENTORY_SHA256=${EXPECTED_SOURCE_INVENTORY_SHA256},EXPECTED_LEDGER_SHA256=${EXPECTED_LEDGER_SHA256},EXPECTED_AUTHORIZATION_SHA256=${AUTHORIZATION_SHA256},EXPECTED_TRAINING_TERMINAL_SHA256=${EXPECTED_TRAINING_TERMINAL_SHA256},EXPECTED_RAW64_TERMINAL_SHA256=${EXPECTED_RAW64_TERMINAL_SHA256},EXPECTED_RAW64_STAGE_SUMMARY_SHA256=${EXPECTED_RAW64_STAGE_SUMMARY_SHA256},LEGACY_PYTHON=${PYTHON}"
JOB_ID="$(sbatch --parsable --array=0-1%2 --export="${common_export}" \
  "${RUN_ROOT}/launchers/planner256_v13.sbatch")"
printf '%s\n' "${JOB_ID}" > "${RUN_ROOT}/status/submitted_planner256_job_id.txt"

export RUN_ROOT JOB_ID AUTHORIZATION_SHA256 EXPECTED_SOURCE_INVENTORY_SHA256
export EXPECTED_SOURCE_ARCHIVE_SHA256 EXPECTED_TRAINING_TERMINAL_SHA256
export EXPECTED_RAW64_TERMINAL_SHA256 EXPECTED_RAW64_STAGE_SUMMARY_SHA256
export EXPECTED_LEDGER_SHA256
"${PYTHON}" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema": "h1_chemistry_first_v13_submission_record_v1",
    "status": "complete",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ["JOB_ID"],
    "array": "0-1%2",
    "arms": ["p0", "sft_v2"],
    "partition": "gpu",
    "source_inventory_sha256": os.environ["EXPECTED_SOURCE_INVENTORY_SHA256"],
    "source_archive_sha256": os.environ["EXPECTED_SOURCE_ARCHIVE_SHA256"],
    "ledger256_sha256": os.environ["EXPECTED_LEDGER_SHA256"],
    "authorization_sha256": os.environ["AUTHORIZATION_SHA256"],
    "training_terminal_sha256": os.environ["EXPECTED_TRAINING_TERMINAL_SHA256"],
    "raw64_terminal_sha256": os.environ["EXPECTED_RAW64_TERMINAL_SHA256"],
    "raw64_stage_summary_sha256": os.environ["EXPECTED_RAW64_STAGE_SUMMARY_SHA256"],
    "diagnostic_user_override": True,
    "generation_math_changed": False,
    "automatic_assembly": False,
    "automatic_downstream": False,
    "automatic_rl": False,
    "smact4_executed_on_a800": False,
}
path = root / "submission_record.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(payload, sort_keys=True))
PY
sha256sum "${RUN_ROOT}/submission_record.json" > "${RUN_ROOT}/submission_record.sha256"
printf 'planner256=%s\n' "${JOB_ID}"
SUBMIT

chmod 500 "${RUN_ROOT}/launchers/planner256_v13.sbatch" \
  "${RUN_ROOT}/launchers/submit_v13_once.sh"
bash -n "${RUN_ROOT}/launchers/planner256_v13.sbatch"
bash -n "${RUN_ROOT}/launchers/submit_v13_once.sh"

export RUN_ROOT EXPECTED_SOURCE_INVENTORY_SHA256 EXPECTED_SOURCE_ARCHIVE_SHA256
export EXPECTED_TRAINING_TERMINAL_SHA256 EXPECTED_RAW64_TERMINAL_SHA256
export EXPECTED_RAW64_STAGE_SUMMARY_SHA256 EXPECTED_LEDGER256_SHA256
"${PYTHON}" - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "h1_chemistry_first_v13_preparation_record_v1",
    "status": "pass",
    "run_root": str(root),
    "source_inventory_sha256": os.environ["EXPECTED_SOURCE_INVENTORY_SHA256"],
    "source_archive_sha256": os.environ["EXPECTED_SOURCE_ARCHIVE_SHA256"],
    "training_terminal_sha256": os.environ["EXPECTED_TRAINING_TERMINAL_SHA256"],
    "raw64_terminal_sha256": os.environ["EXPECTED_RAW64_TERMINAL_SHA256"],
    "raw64_stage_summary_sha256": os.environ["EXPECTED_RAW64_STAGE_SUMMARY_SHA256"],
    "ledger256_sha256": os.environ["EXPECTED_LEDGER256_SHA256"],
    "authorization_sha256": sha(root / "AUTHORIZATION.json"),
    "launcher_sha256": {
        path.name: sha(path) for path in sorted((root / "launchers").iterdir())
    },
    "arms": ["p0", "sft_v2"],
    "sft_v2_c_excluded_as_dominated": True,
    "broad_tests_repeated": False,
    "generation": False,
    "smact4_executed_on_a800": False,
}
(root / "V13_PREPARATION_RECORD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

tar -czf "${RUN_ROOT}/launcher_archive.tar.gz" -C "${RUN_ROOT}" \
  AUTHORIZATION.json LEDGER256.json launchers V13_PREPARATION_RECORD.json
(
  cd "${RUN_ROOT}"
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum AUTHORIZATION.json LEDGER256.json V13_PREPARATION_RECORD.json \
    launcher_archive.tar.gz prepare_v13_user_override_diagnostic_raw256_on_a800.sh
) > "${RUN_ROOT}/V13_SHA256.txt"
(cd "${RUN_ROOT}" && sha256sum -c V13_SHA256.txt)
touch "${RUN_ROOT}/status/preparation_SUCCESS"
chmod 400 "${RUN_ROOT}/AUTHORIZATION.json" "${RUN_ROOT}/LEDGER256.json" \
  "${RUN_ROOT}/V13_PREPARATION_RECORD.json" "${RUN_ROOT}/V13_SHA256.txt" \
  "${RUN_ROOT}/launcher_archive.tar.gz" \
  "${RUN_ROOT}/prepare_v13_user_override_diagnostic_raw256_on_a800.sh"
cat "${RUN_ROOT}/V13_PREPARATION_RECORD.json"

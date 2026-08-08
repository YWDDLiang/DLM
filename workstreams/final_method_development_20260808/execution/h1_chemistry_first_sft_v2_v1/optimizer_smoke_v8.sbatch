#!/bin/bash
#SBATCH --job-name=h1-cf-opt2-v8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8/logs/%A_%a_optimizer_smoke.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8/logs/%A_%a_optimizer_smoke.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_optimizer_zero_lr_audit_repair_v8"
SOURCE_ROOT="${RUN_ROOT}/source"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${LEGACY_PYTHON:?missing training runtime}"
: "${SLURM_ARRAY_TASK_ID:?missing candidate identity}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) CANDIDATE=sft_v2 ;;
  1) CANDIDATE=sft_v2_c ;;
  *) exit 3 ;;
esac
test "${SLURM_JOB_PARTITION:-}" = gpu
test -f "${RUN_ROOT}/status/data_SUCCESS"
test ! -e "${RUN_ROOT}/optimizer_smoke/${CANDIDATE}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/optimizer_smoke" "${RUN_ROOT}/status"
on_error() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/status/optimizer_smoke_${CANDIDATE}_exit_code.txt"
  touch "${RUN_ROOT}/status/optimizer_smoke_${CANDIDATE}_FAILED"
  exit "${rc}"
}
trap on_error ERR

"${LEGACY_PYTHON}" - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0] or not torch.cuda.is_bf16_supported():
    raise SystemExit(f"expected one BF16 A800, found {names}")
print({"gpu": names, "torch": torch.__version__}, flush=True)
PY

"${LEGACY_PYTHON}" scripts/llama_h1_chemistry_first_sft.py \
  --model-path "${MODEL_PATH}" --p0-adapter-path "${P0_ADAPTER}" \
  --data-dir "${RUN_ROOT}/data/sft/${CANDIDATE}" \
  --data-audit-report "${RUN_ROOT}/data/sft/audit_report.json" \
  --output-dir "${RUN_ROOT}/optimizer_smoke/${CANDIDATE}" \
  --candidate "${CANDIDATE}" --max-length 512 --batch-size 1 \
  --grad-accum 8 --lr 0.000002 --weight-decay 0 --seed 26080817 \
  --logging-steps 1 --optimizer-smoke-updates 2

report="${RUN_ROOT}/optimizer_smoke/${CANDIDATE}/optimizer_smoke_report.json"
test -f "${report}"
test -f "${RUN_ROOT}/optimizer_smoke/${CANDIDATE}/_SUCCESS"
"${LEGACY_PYTHON}" - "${report}" "${CANDIDATE}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
audits = report.get("optimizer_step_audits") or []
if not (
    report.get("schema") == "h1_chemistry_first_two_step_optimizer_smoke_v1"
    and report.get("status") == "pass"
    and report.get("candidate") == sys.argv[2]
    and report.get("full_training_total_updates") == 4505
    and report.get("full_training_warmup_steps") == 135
    and report.get("optimizer_updates") == 2
    and report.get("microbatch_count") == 16
    and report.get("scientific_checkpoint_saved") is False
    and report.get("generation") is False
    and report.get("smact4_executed_on_a800") is False
    and report.get("failures") == []
    and len(audits) == 2
    and all(item.get("passed") is True for item in audits)
    and audits[0].get("lr_used") == 0.0
    and audits[0].get("candidate_changed") is False
    and audits[0].get("optimizer_state_after", {}).get("expected_step") == 1
    and audits[1].get("candidate_changed") is True
    and audits[1].get("optimizer_state_after", {}).get("expected_step") == 2
):
    raise SystemExit("two-step optimizer smoke report is not admissible")
PY
sha256sum "${report}" > "${RUN_ROOT}/optimizer_smoke/${CANDIDATE}/optimizer_smoke_report.sha256"
printf '0\n' > "${RUN_ROOT}/status/optimizer_smoke_${CANDIDATE}_exit_code.txt"
touch "${RUN_ROOT}/status/optimizer_smoke_${CANDIDATE}_SUCCESS"
echo "H1_CHEMISTRY_FIRST_OPTIMIZER_SMOKE_PASS candidate=${CANDIDATE}"

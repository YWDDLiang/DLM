#!/bin/bash
#SBATCH --job-name=h1-cf-train-v2
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5/logs/%A_%a_train.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5/logs/%A_%a_train.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5"
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
test "${SLURM_JOB_PARTITION:-}" = gpu_long
test -f "${RUN_ROOT}/status/smoke_${CANDIDATE}_SUCCESS"
test ! -e "${RUN_ROOT}/training/${CANDIDATE}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/training" "${RUN_ROOT}/status"

set +e
"${LEGACY_PYTHON}" scripts/llama_h1_chemistry_first_sft.py \
  --model-path "${MODEL_PATH}" --p0-adapter-path "${P0_ADAPTER}" \
  --data-dir "${RUN_ROOT}/data/sft/${CANDIDATE}" \
  --data-audit-report "${RUN_ROOT}/data/sft/audit_report.json" \
  --output-dir "${RUN_ROOT}/training/${CANDIDATE}" \
  --candidate "${CANDIDATE}" --max-length 512 --batch-size 1 \
  --grad-accum 8 --lr 0.000002 --weight-decay 0 --seed 26080817 \
  --logging-steps 10
trainer_rc=$?
set -e

terminal="${RUN_ROOT}/training/${CANDIDATE}/terminal_report.json"
if test -f "${terminal}"; then
  terminal_status="$("${LEGACY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${terminal}")"
else
  terminal_status=missing
fi
case "${terminal_status}" in
  pass)
    test "${trainer_rc}" -eq 0
    test -f "${RUN_ROOT}/training/${CANDIDATE}/_SUCCESS"
    printf '0\n' > "${RUN_ROOT}/status/train_${CANDIDATE}_exit_code.txt"
    touch "${RUN_ROOT}/status/train_${CANDIDATE}_SUCCESS"
    echo "H1_CHEMISTRY_FIRST_TRAIN_PASS candidate=${CANDIDATE}"
    ;;
  anchor_nll_noninferiority_failure)
    test "${trainer_rc}" -ne 0
    test -f "${RUN_ROOT}/training/${CANDIDATE}/_FAILED"
    printf '0\n' > "${RUN_ROOT}/status/train_${CANDIDATE}_exit_code.txt"
    touch "${RUN_ROOT}/status/train_${CANDIDATE}_SCIENTIFIC_STOP"
    echo "H1_CHEMISTRY_FIRST_TRAIN_SCIENTIFIC_STOP candidate=${CANDIDATE}"
    ;;
  *)
    if test "${trainer_rc}" -eq 0; then trainer_rc=2; fi
    printf '%s\n' "${trainer_rc}" > "${RUN_ROOT}/status/train_${CANDIDATE}_exit_code.txt"
    touch "${RUN_ROOT}/status/train_${CANDIDATE}_FAILED"
    echo "H1_CHEMISTRY_FIRST_TRAIN_ENGINEERING_FAILURE candidate=${CANDIDATE} status=${terminal_status}" >&2
    exit "${trainer_rc}"
    ;;
esac

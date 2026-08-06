#!/usr/bin/env bash
#SBATCH --job-name=pg-g1-sample-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_plangraph_dlm_g1_sampling_repair_v2/slurm/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_plangraph_dlm_g1_sampling_repair_v2/slurm/%x-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260801_plangraph_dlm_g1_sampling_repair_v2"
SOURCE_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/g1_source_v2"
EXECUTION_ROOT="${PROJECT_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution"
MANIFEST="${EXECUTION_ROOT}/G1_EXECUTION_MANIFEST_V2.json"
AUTHORIZATION="${EXECUTION_ROOT}/G1_AUTHORIZATION_V2.json"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
INITIAL_ADAPTER="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
INITIAL_ADAPTER_SHA256=65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${G1_EXECUTION_MANIFEST_SHA256:?missing G1_EXECUTION_MANIFEST_SHA256}"
: "${G1_AUTHORIZATION_SHA256:?missing G1_AUTHORIZATION_SHA256}"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    ARM=P0
    CHECKPOINT="${INITIAL_ADAPTER}"
    CHECKPOINT_IDENTITY="${INITIAL_ADAPTER_SHA256}"
    ;;
  1)
    ARM=PG
    TRAINING_REPORT="${RUN_ROOT}/training/${ARM}/training_report.json"
    CHECKPOINT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_dir"])' "${TRAINING_REPORT}")"
    CHECKPOINT_IDENTITY="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_manifest_sha256"])' "${TRAINING_REPORT}")"
    ;;
  2)
    ARM=PG-shuffle
    TRAINING_REPORT="${RUN_ROOT}/training/${ARM}/training_report.json"
    CHECKPOINT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_dir"])' "${TRAINING_REPORT}")"
    CHECKPOINT_IDENTITY="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_manifest_sha256"])' "${TRAINING_REPORT}")"
    ;;
  *) echo "unexpected G1 sampling arm" >&2; exit 2 ;;
esac
OUTPUT_DIR="${RUN_ROOT}/sampling/${ARM}"
LEDGER="${RUN_ROOT}/data/ledger/seed_ledger.jsonl"
LEDGER_SHA="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed_ledger_sha256"])' "${RUN_ROOT}/data/ledger_report.json")"

trap 'status=$?; printf "%s\n" "${status}" > "${RUN_ROOT}/status/sample_${ARM}_exit_status.txt"; nvidia-smi > "${RUN_ROOT}/status/sample_${ARM}_gpu_end.txt" 2>&1 || true; exit "${status}"' EXIT
test "${SLURM_CPUS_PER_TASK:-0}" -le 8
test "${SLURM_ARRAY_TASK_COUNT:-0}" -eq 3
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${G1_EXECUTION_MANIFEST_SHA256}"
test "$(sha256sum "${AUTHORIZATION}" | awk '{print $1}')" = "${G1_AUTHORIZATION_SHA256}"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_SHA256.txt)
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test -s "${LEDGER}"
test ! -e "${OUTPUT_DIR}"

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/sample_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN

"${PYTHON}" -c 'import os,torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; assert len(names)==1 and "A800" in names[0] and int(os.environ.get("SLURM_CPUS_PER_TASK","99"))<=8, names; print({"gpu":names})'
nvidia-smi > "${RUN_ROOT}/status/sample_${ARM}_gpu_start.txt"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader,nounits -l 2 > "${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.gpu.csv" &
MONITOR_PID=$!
cleanup_monitor() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap 'status=$?; cleanup_monitor; printf "%s\n" "${status}" > "${RUN_ROOT}/status/sample_'"${ARM}"'_exit_status.txt"; nvidia-smi > "${RUN_ROOT}/status/sample_'"${ARM}"'_gpu_end.txt" 2>&1 || true; exit "${status}"' EXIT

"${PYTHON}" "${SOURCE_ROOT}/sample_g1_planner.py" \
  --arm "${ARM}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --checkpoint-identity-sha256 "${CHECKPOINT_IDENTITY}" \
  --ledger "${LEDGER}" \
  --ledger-sha256 "${LEDGER_SHA}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --max-new-tokens 640 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50

test -s "${OUTPUT_DIR}/planner_report.json"
echo "G1_SAMPLE_ACCEPTANCE=PASS ARM=${ARM}"

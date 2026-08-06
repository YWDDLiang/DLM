#!/bin/bash
#SBATCH --job-name=h1-crplan-cr0-v5
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_script_package_repair_v5/logs/%j_cr0.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_script_package_repair_v5/logs/%j_cr0.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260804_h1_crplan_r0_paired32_script_package_repair_v5"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_r0_paired32_script_package_repair_v5"
MODEL_PATH=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
CHECKPOINT="${PROJECT_ROOT}/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

phase() {
  printf '{"event":"h1_crplan_cr0_shell_phase","phase":"%s","utc":"%s"}\n' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

phase job_enter
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing source manifest identity}"
: "${EXPECTED_ADAPTER_SHA256:?missing adapter identity}"

test "${SLURM_JOB_PARTITION:-}" = normal
test "${SLURM_CPUS_PER_TASK:-0}" -eq 4
test -d "${SOURCE_ROOT}"
test -f "${SOURCE_ROOT}/scripts/__init__.py"
test -d "${MODEL_PATH}"
test -d "${CHECKPOINT}"
test "$(sha256sum "${CHECKPOINT}/adapter_model.safetensors" | cut -d' ' -f1)" = "${EXPECTED_ADAPTER_SHA256}"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test ! -e "${RUN_ROOT}/cr0"
phase asset_identity_complete

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt
phase source_manifest_verification_complete

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u
phase conda_activation_complete

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# CR-0 only needs the frozen tokenizer.  Transformers 4.54 otherwise probes
# every installed ML backend while importing AutoTokenizer, which can spend
# the complete scheduler envelope reading package metadata from shared storage.
# The paired GPU job intentionally does not inherit these CR-0-only settings.
export USE_TORCH=0
export USE_TF=0
export USE_FLAX=0
export USE_TORCH_XLA=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=
phase tokenizer_only_backend_contract_configured

mkdir -p "${RUN_ROOT}/cr0"
phase unit_tests_start
"${PYTHON}" -m unittest \
  tests.test_h1_crplan \
  tests.test_h1_crplan_cr0_runtime \
  -v \
  >"${RUN_ROOT}/cr0/unit_tests.out" \
  2>"${RUN_ROOT}/cr0/unit_tests.err"
phase unit_tests_complete
phase audit_process_start
"${PYTHON}" "${EXECUTION_DIR}/run_cr0_audit.py" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT}" \
  --source-manifest "${SOURCE_ROOT}/SOURCE_SHA256.txt" \
  --output "${RUN_ROOT}/cr0/terminal_report.json"
phase audit_process_complete

touch "${RUN_ROOT}/cr0/_SUCCESS"
phase success_marker_written
echo "H1_CRPLAN_CR0_COMPLETE"

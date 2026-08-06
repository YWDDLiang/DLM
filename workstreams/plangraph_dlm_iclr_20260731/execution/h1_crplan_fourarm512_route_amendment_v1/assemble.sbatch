#!/bin/bash
#SBATCH --job-name=h1-cr-4a512-asm-v1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1/logs/%j_assembly.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1/logs/%j_assembly.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260805_h1_crplan_fourarm512_route_amendment_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_crplan_fourarm512_route_amendment_v1"
V4_TERMINAL="${PROJECT_ROOT}/runs/20260805_h1_crplan_fourarm512_exact_support_optimization_preflight_runtime_repair_v4/preflight/terminal_report.json"
E1_TERMINAL="${PROJECT_ROOT}/runs/20260805_h1_crplan_e1_physical_performance_probe_v1/terminal/terminal_report.json"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

: "${EXPECTED_SOURCE_MANIFEST_SHA256:?missing source manifest identity}"
: "${EXPECTED_LEDGER_SHA256:?missing science ledger identity}"
: "${ARRAY_JOB_ID:?missing array job identity}"

test "${SLURM_JOB_PARTITION:-}" = normal
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test -x "${PYTHON}"
test ! -e "${RUN_ROOT}/terminal"
test ! -e "${RUN_ROOT}/_SUCCESS"
test ! -e "${RUN_ROOT}/_SCIENTIFIC_STOP"
test ! -e "${RUN_ROOT}/_FAILED"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
test "$(sha256sum "${EXECUTION_DIR}/SCIENCE_LEDGER.json" | cut -d' ' -f1)" = "${EXPECTED_LEDGER_SHA256}"

cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

set +u
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
set -u

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "${RUN_ROOT}/terminal" "${RUN_ROOT}/status"
set +e
"${PYTHON}" "${EXECUTION_DIR}/evaluate_fourarm512.py" \
  --run-root "${RUN_ROOT}" \
  --config "${EXECUTION_DIR}/CONFIG.json" \
  --ledger "${EXECUTION_DIR}/SCIENCE_LEDGER.json" \
  --source-manifest "${SOURCE_ROOT}/SOURCE_SHA256.txt" \
  --v4-terminal "${V4_TERMINAL}" \
  --e1-terminal "${E1_TERMINAL}" \
  --array-job-id "${ARRAY_JOB_ID}" \
  --output "${RUN_ROOT}/terminal/terminal_report.json"
evaluator_rc=$?
set -e
printf '%s\n' "${evaluator_rc}" > "${RUN_ROOT}/status/assembly_evaluator_exit_code.txt"

if test -f "${RUN_ROOT}/terminal/terminal_report.json"; then
  sha256sum "${RUN_ROOT}/terminal/terminal_report.json" \
    > "${RUN_ROOT}/terminal/terminal_report.sha256"
fi
if test "${evaluator_rc}" -ne 0; then
  touch "${RUN_ROOT}/_FAILED"
  echo "H1_CRPLAN_FOURARM512_ENGINEERING_FAILURE" >&2
  exit "${evaluator_rc}"
fi

terminal_status="$(
  "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${RUN_ROOT}/terminal/terminal_report.json"
)"
case "${terminal_status}" in
  mechanism_gate_pass)
    touch "${RUN_ROOT}/_SUCCESS"
    echo "H1_CRPLAN_FOURARM512_MECHANISM_GATE_PASS"
    ;;
  scientific_stop)
    touch "${RUN_ROOT}/_SCIENTIFIC_STOP"
    echo "H1_CRPLAN_FOURARM512_SCIENTIFIC_STOP"
    ;;
  *)
    touch "${RUN_ROOT}/_FAILED"
    echo "unexpected terminal status ${terminal_status}" >&2
    exit 2
    ;;
esac
exit 0

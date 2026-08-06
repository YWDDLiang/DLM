#!/bin/bash
#SBATCH --job-name=h1-nc-a256-v1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%j_assemble256.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260806_h1_nocharge_ion_aux_sft_v1/logs/%j_assemble256.err

set -Eeuo pipefail
umask 077
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260806_h1_nocharge_ion_aux_sft_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/h1_nocharge_ion_aux_sft_v1"
STAGE_ROOT="${RUN_ROOT}/planner256"
TERMINAL="${STAGE_ROOT}/terminal/terminal_report.json"

: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${EXPECTED_LEDGER_SHA256:?missing planner256 ledger identity}"
: "${LEGACY_PYTHON:?missing legacy evaluator runtime}"
: "${SMACT4_PYTHON:?missing SMACT4 runtime}"
test "${SLURM_JOB_PARTITION:-}" = normal
test -f "${RUN_ROOT}/planner64/_SUCCESS"
test ! -e "${STAGE_ROOT}/terminal"
test ! -e "${STAGE_ROOT}/_SUCCESS"
test ! -e "${STAGE_ROOT}/_SCIENTIFIC_STOP"
test ! -e "${STAGE_ROOT}/_FAILED"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${EXECUTION_DIR}/LEDGER256.json" | cut -d' ' -f1)" = "${EXPECTED_LEDGER_SHA256}"
cd "${SOURCE_ROOT}"
sha256sum -c SOURCE_SHA256.txt

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/.pycache/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${STAGE_ROOT}/terminal" "${STAGE_ROOT}/secondary" "${RUN_ROOT}/status"

fail_terminal() {
  local reason="$1"
  if ! test -f "${TERMINAL}"; then
    "${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_stage_engineering_failure.py" \
      --stage 256 --output "${TERMINAL}" --reason "${reason}"
  fi
  sha256sum "${TERMINAL}" > "${STAGE_ROOT}/terminal/terminal_report.sha256"
  touch "${STAGE_ROOT}/_FAILED"
  printf '2\n' > "${RUN_ROOT}/status/assemble256_exit_code.txt"
  echo "H1_NOCHARGE_PLANNER256_ENGINEERING_FAILURE reason=${reason}" >&2
  exit 2
}

for arm in p0 c0 c1; do
  test -f "${RUN_ROOT}/status/planner256_${arm}_SUCCESS" || fail_terminal "missing_planner256_${arm}_SUCCESS"
  test "$(wc -l < "${STAGE_ROOT}/${arm}/raw_generations.jsonl")" -eq 256 || fail_terminal "raw_count_${arm}"
  set +e
  "${SMACT4_PYTHON}" scripts/audit_h1_nocharge_planner_smact4.py \
    --raw-generations-jsonl "${STAGE_ROOT}/${arm}/raw_generations.jsonl" \
    --denominator 256 --arm "${arm}" \
    --output-json "${STAGE_ROOT}/secondary/${arm}_smact4.json"
  audit_rc=$?
  set -e
  test "${audit_rc}" -eq 0 || fail_terminal "smact4_audit_${arm}_exit_${audit_rc}"
done

set +e
"${LEGACY_PYTHON}" scripts/evaluate_h1_nocharge_sft_planner_gate.py \
  --stage 256 --ledger "${EXECUTION_DIR}/LEDGER256.json" \
  --p0-raw "${STAGE_ROOT}/p0/raw_generations.jsonl" \
  --c0-raw "${STAGE_ROOT}/c0/raw_generations.jsonl" \
  --c1-raw "${STAGE_ROOT}/c1/raw_generations.jsonl" \
  --p0-smact4 "${STAGE_ROOT}/secondary/p0_smact4.json" \
  --c0-smact4 "${STAGE_ROOT}/secondary/c0_smact4.json" \
  --c1-smact4 "${STAGE_ROOT}/secondary/c1_smact4.json" \
  --output-json "${TERMINAL}" \
  --output-md "${STAGE_ROOT}/terminal/terminal_report.md"
evaluator_rc=$?
set -e
test -f "${TERMINAL}" || fail_terminal "legacy_evaluator_exit_${evaluator_rc}_without_report"
sha256sum "${TERMINAL}" > "${STAGE_ROOT}/terminal/terminal_report.sha256"
printf '%s\n' "${evaluator_rc}" > "${RUN_ROOT}/status/assemble256_exit_code.txt"
status="$("${LEGACY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${TERMINAL}")"
case "${status}" in
  planner_gate_pass)
    test "${evaluator_rc}" -eq 0 || fail_terminal "pass_report_nonzero_exit_${evaluator_rc}"
    touch "${STAGE_ROOT}/_SUCCESS"
    echo H1_NOCHARGE_PLANNER256_GATE_PASS
    ;;
  scientific_stop)
    test "${evaluator_rc}" -eq 0 || fail_terminal "scientific_stop_nonzero_exit_${evaluator_rc}"
    touch "${STAGE_ROOT}/_SCIENTIFIC_STOP"
    echo H1_NOCHARGE_PLANNER256_SCIENTIFIC_STOP
    ;;
  engineering_failure)
    touch "${STAGE_ROOT}/_FAILED"
    echo H1_NOCHARGE_PLANNER256_ENGINEERING_FAILURE >&2
    exit 2
    ;;
  *) fail_terminal "unexpected_terminal_status_${status}" ;;
esac

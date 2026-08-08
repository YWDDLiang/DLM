#!/bin/bash
# This file is a template invoked through the stage-specific wrappers.
# It is not submitted directly.
set -Eeuo pipefail
umask 077

: "${STAGE:?missing stage}"
: "${EXPECTED_CANDIDATES:?missing expected candidates}"
: "${EXPECTED_SOURCE_INVENTORY_SHA256:?missing source identity}"
: "${EXPECTED_LEDGER_SHA256:?missing ledger identity}"
: "${LEGACY_PYTHON:?missing legacy runtime}"
: "${SMACT4_PYTHON:?missing exact SMACT4 runtime}"
case "${STAGE}" in 64|256) ;; *) exit 3 ;; esac

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_v1"
SOURCE_ROOT="${RUN_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1"
STAGE_ROOT="${RUN_ROOT}/planner${STAGE}"
LEDGER="${EXECUTION_DIR}/LEDGER${STAGE}.json"

test "${SLURM_JOB_PARTITION:-}" = normal
test ! -e "${STAGE_ROOT}/terminal"
test ! -e "${STAGE_ROOT}/_SUCCESS"
test ! -e "${STAGE_ROOT}/_SCIENTIFIC_STOP"
test ! -e "${STAGE_ROOT}/_FAILED"
test "$(sha256sum "${SOURCE_ROOT}/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "${EXPECTED_SOURCE_INVENTORY_SHA256}"
test "$(sha256sum "${LEDGER}" | cut -d' ' -f1)" = "${EXPECTED_LEDGER_SHA256}"
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

write_engineering_failure() {
  local candidate="$1"
  local reason="$2"
  local terminal="${STAGE_ROOT}/terminal/${candidate}_terminal_report.json"
  if ! test -e "${terminal}"; then
    "${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_stage_engineering_failure.py" \
      --stage "${STAGE}" --candidate "${candidate}" --reason "${reason}" \
      --output "${terminal}"
  fi
}

IFS=',' read -r -a CANDIDATES <<< "${EXPECTED_CANDIDATES}"
test "${#CANDIDATES[@]}" -ge 1
for candidate in "${CANDIDATES[@]}"; do
  case "${candidate}" in sft_v2|sft_v2_c) ;; *) exit 3 ;; esac
done

p0_ready=true
if ! test -f "${RUN_ROOT}/status/planner${STAGE}_p0_SUCCESS" || \
   ! test "$(wc -l < "${STAGE_ROOT}/p0/raw_generations.jsonl" 2>/dev/null || printf 0)" -eq "${STAGE}"; then
  p0_ready=false
else
  set +e
  "${SMACT4_PYTHON}" scripts/audit_h1_chemistry_first_planner_smact4.py \
    --raw-generations-jsonl "${STAGE_ROOT}/p0/raw_generations.jsonl" \
    --denominator "${STAGE}" --arm p0 \
    --output-json "${STAGE_ROOT}/secondary/p0_smact4.json"
  p0_audit_rc=$?
  set -e
  test "${p0_audit_rc}" -eq 0 || p0_ready=false
fi

for candidate in "${CANDIDATES[@]}"; do
  terminal="${STAGE_ROOT}/terminal/${candidate}_terminal_report.json"
  if test "${STAGE}" -eq 64 && test -f "${RUN_ROOT}/status/train_${candidate}_SCIENTIFIC_STOP"; then
    "${LEGACY_PYTHON}" "${EXECUTION_DIR}/write_stage_training_stop.py" \
      --stage 64 --candidate "${candidate}" \
      --training-terminal "${RUN_ROOT}/training/${candidate}/terminal_report.json" \
      --output "${terminal}"
    sha256sum "${terminal}" > "${terminal}.sha256"
    continue
  fi
  if test "${p0_ready}" != true; then
    write_engineering_failure "${candidate}" "p0_generation_or_smact4_audit_failure"
    sha256sum "${terminal}" > "${terminal}.sha256"
    continue
  fi
  if ! test -f "${RUN_ROOT}/status/planner${STAGE}_${candidate}_SUCCESS" || \
     ! test "$(wc -l < "${STAGE_ROOT}/${candidate}/raw_generations.jsonl" 2>/dev/null || printf 0)" -eq "${STAGE}"; then
    write_engineering_failure "${candidate}" "candidate_generation_failure"
    sha256sum "${terminal}" > "${terminal}.sha256"
    continue
  fi
  set +e
  "${SMACT4_PYTHON}" scripts/audit_h1_chemistry_first_planner_smact4.py \
    --raw-generations-jsonl "${STAGE_ROOT}/${candidate}/raw_generations.jsonl" \
    --denominator "${STAGE}" --arm "${candidate}" \
    --output-json "${STAGE_ROOT}/secondary/${candidate}_smact4.json"
  audit_rc=$?
  set -e
  if test "${audit_rc}" -ne 0; then
    write_engineering_failure "${candidate}" "candidate_smact4_audit_exit_${audit_rc}"
    sha256sum "${terminal}" > "${terminal}.sha256"
    continue
  fi
  set +e
  "${LEGACY_PYTHON}" scripts/evaluate_h1_chemistry_first_planner_gate.py \
    --stage "${STAGE}" --candidate-id "${candidate}" --ledger "${LEDGER}" \
    --training-terminal "${RUN_ROOT}/training/${candidate}/terminal_report.json" \
    --p0-raw "${STAGE_ROOT}/p0/raw_generations.jsonl" \
    --p0-smact4 "${STAGE_ROOT}/secondary/p0_smact4.json" \
    --candidate-raw "${STAGE_ROOT}/${candidate}/raw_generations.jsonl" \
    --candidate-smact4 "${STAGE_ROOT}/secondary/${candidate}_smact4.json" \
    --output-json "${terminal}" \
    --output-md "${STAGE_ROOT}/terminal/${candidate}_terminal_report.md"
  evaluator_rc=$?
  set -e
  if ! test -f "${terminal}"; then
    write_engineering_failure "${candidate}" "legacy_evaluator_exit_${evaluator_rc}_without_report"
  fi
  sha256sum "${terminal}" > "${terminal}.sha256"
done

"${LEGACY_PYTHON}" "${EXECUTION_DIR}/assemble_stage_summary.py" \
  --stage "${STAGE}" --stage-root "${STAGE_ROOT}" \
  --candidates "${EXPECTED_CANDIDATES}" \
  --output "${STAGE_ROOT}/terminal/stage_summary.json"
sha256sum "${STAGE_ROOT}/terminal/stage_summary.json" > "${STAGE_ROOT}/terminal/stage_summary.sha256"
stage_status="$("${LEGACY_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${STAGE_ROOT}/terminal/stage_summary.json")"
printf '0\n' > "${RUN_ROOT}/status/assemble${STAGE}_exit_code.txt"
case "${stage_status}" in
  planner_gate_pass)
    touch "${STAGE_ROOT}/_SUCCESS"
    echo "H1_CHEMISTRY_FIRST_PLANNER${STAGE}_GATE_PASS"
    ;;
  scientific_stop)
    touch "${STAGE_ROOT}/_SCIENTIFIC_STOP"
    echo "H1_CHEMISTRY_FIRST_PLANNER${STAGE}_SCIENTIFIC_STOP"
    ;;
  engineering_failure)
    touch "${STAGE_ROOT}/_FAILED"
    printf '2\n' > "${RUN_ROOT}/status/assemble${STAGE}_exit_code.txt"
    echo "H1_CHEMISTRY_FIRST_PLANNER${STAGE}_ENGINEERING_FAILURE" >&2
    exit 2
    ;;
  *) exit 3 ;;
esac

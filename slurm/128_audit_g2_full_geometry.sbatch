#!/usr/bin/env bash
#SBATCH --job-name=g2-full-geom-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=g2-full-geom-audit-%j.out
#SBATCH --error=g2-full-geom-audit-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly AUDITOR="${CODE_ROOT}/scripts/audit_g2_full_geometry_contract.py"
readonly AUDITOR_SHA="${G2_FULL_AUDITOR_SHA256:?pin auditor SHA256}"
readonly OPS="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_ops.py"
readonly OPS_SHA="${G2_FULL_OPS_SHA256:?pin shared PBC/radius SHA256}"
readonly TRAIN="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831/train.jsonl"
readonly BASE_RAW="${ROOT}/runs/compact_v2_periodic_relation_g2_final_generation_39137/arm_BASE/body/raw_generations.jsonl"
readonly G2_RAW="${ROOT}/runs/compact_v2_periodic_relation_g2_final_generation_39137/arm_G2/body/raw_generations.jsonl"
readonly RUN="${ROOT}/runs/g2_full_geometry_contract_audit_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
[[ "$(sha256sum "${AUDITOR}" | awk '{print $1}')" == "${AUDITOR_SHA}" ]]
[[ "$(sha256sum "${OPS}" | awk '{print $1}')" == "${OPS_SHA}" ]]

"${PYTHON}" "${AUDITOR}" \
  --train-jsonl "${TRAIN}" \
  --generation-jsonl "${BASE_RAW}" \
  --generation-jsonl "${G2_RAW}" \
  --output-dir "${RUN}" \
  --margin-scale 0.55 \
  --margin-floor 0.60 \
  --margin-ceiling 1.40

[[ -f "${RUN}/_SUCCESS" && ! -e "${RUN}/_FAILED" ]]
sha256sum "${RUN}/GEOMETRY_CONTRACT_AUDIT.json" >"${RUN}/OUTPUTS.sha256"
printf '%s\n' "${RUN}"

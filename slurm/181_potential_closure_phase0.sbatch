#!/usr/bin/env bash
#SBATCH --job-name=potential-closure-p0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=potential-closure-p0-%j.out
#SBATCH --error=potential-closure-p0-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_spade_b14c457}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly MP20_VAL="${PROJECT}/reference/crysllmgen/data/mp_20/val.csv"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly SNAPSHOT="${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_phase0_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/audit"

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${SOURCE}/scripts/audit_potential_closure_phase0.py" \
  "${SOURCE}/scripts/run_direct_validity_fast.py" "${MP20_VAL}" \
  "${SNAPSHOT}/eval_utils.py"; do
  [[ -f "${path}" ]]
done

cat >"${RUN}/contract.tsv" <<EOF
field	value
scope	MP20-val continuous versus exact deployed 7+4N round trip
rows	9047
fast_validity_subset	512
device	job-local cuda:0
chgnet_batch	16
official_query	false
formal_training	false
EOF

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/audit_potential_closure_phase0.py" \
  --validation-csv "${MP20_VAL}" \
  --snapshot-root "${SNAPSHOT}" \
  --output-dir "${OUTPUT}" \
  --device cuda:0 --batch-size 16 --workers 8 \
  --expected-rows 9047 --fast-subset-size 512 --selection-seed 20260904 \
  --median-energy-limit 0.015 --validity-drop-limit 0.01 \
  --retention-limit 0.60 \
  >"${RUN}/phase0.out" 2>"${RUN}/phase0.err"

[[ -f "${OUTPUT}/PHASE0_REPORT.json" ]]
[[ -f "${OUTPUT}/_SUCCESS" || -f "${OUTPUT}/_BLOCKED" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

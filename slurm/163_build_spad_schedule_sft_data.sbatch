#!/usr/bin/env bash
#SBATCH --job-name=spad-sft-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=spad-sft-data-%j.out
#SBATCH --error=spad-sft-data-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly TEACHER="${ROOT}/data/c3fd_native_teacher_sft_canonical_v2_20260902"
readonly POINTER="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly OUTPUT="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly RUN="${ROOT}/runs/build_spad_schedule_sft_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ ! -e "${OUTPUT}" ]]
env CUDA_VISIBLE_DEVICES= PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/scripts/build_spad_schedule_sft_data.py" \
  --teacher-data-dir "${TEACHER}" --pointer-data-dir "${POINTER}" \
  --output-dir "${OUTPUT}" --seed 20260903 \
  >"${RUN}/build.out" 2>"${RUN}/build.err"
[[ -f "${OUTPUT}/_SUCCESS" ]]
"${PYTHON}" - "${OUTPUT}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); m=json.load(open(r/'manifest.json'))
assert m['splits']['train']['rows']==27136 and m['splits']['val']['rows']==9047
assert set(m['splits']['train']['mask_classes'])=={
 'deterministic_random_geometry','program_predictor','suffix_visible_anchor_correction'}
assert m['energy_hull_chgnet_model494_read'] is False
PY
trap - ERR
touch "${RUN}/_SUCCESS"

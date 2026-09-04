#!/usr/bin/env bash
#SBATCH --job-name=spad-tau800
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=spad-tau800-%j.out
#SBATCH --error=spad-tau800-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly NATIVE_RUN="${ROOT}/runs/spad_basin_closure_native_s17_39704"
readonly BODY="${NATIVE_RUN}/closure_ce/stream17/body"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_tau800_s17_${SLURM_JOB_ID}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${NATIVE_RUN}/_SUCCESS" "${BODY}/_SUCCESS" "${BODY}/proposal_graphs.pt" "${REFINER}"; do [[ -f "$p" ]]; done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#allocated[@]}" -eq 1 ]]
env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
 "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
 --proposal-graphs "${BODY}/proposal_graphs.pt" --checkpoint "${REFINER}" \
 --output-dir "${RUN}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
 --seed 101117 --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
 >"${RUN}/logs/refine.out" 2>"${RUN}/logs/refine.err"
"${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
 --body-dir "${BODY}" --refine-dir "${RUN}/refine" --output-dir "${RUN}/generation" \
 --arm candidate --repeat 0 --dlm-seed 91117 --refiner-seed 101117 --denominator 256
[[ "$(wc -l <"${RUN}/generation/generation.jsonl")" -eq 256 ]]
trap - ERR; touch "${RUN}/_SUCCESS"

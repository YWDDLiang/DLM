#!/usr/bin/env bash
#SBATCH --job-name=plan1200-tau800
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=05:00:00
#SBATCH --output=plan1200-tau800-%j.out
#SBATCH --error=plan1200-tau800-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${ROOT}/runs/g2_known1200_baseline_fast_39183"
readonly RUN="${ROOT}/runs/plan1200_tau800_${SLURM_JOB_ID}"
readonly PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly REF=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main

mkdir "${RUN}" "${RUN}/logs"
trap 'set +e; touch "${RUN}/_FAILED"; exit 1' ERR
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu[@]}" -eq 2 ]]

PYTHONPATH="${CODE}/src" "${PY}" "${CODE}/scripts/combine_known1200_proposals.py" \
  --baseline-run "${BASE}" --output-dir "${RUN}/combined"
env CUDA_VISIBLE_DEVICES="${gpu[0]},${gpu[1]}" "${TORCHRUN}" --standalone \
  --master-port 29643 --nproc_per_node=2 \
  "${CODE}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${RUN}/combined/proposal_graphs.pt" --checkpoint "${REF}" \
  --output-dir "${RUN}/tau800/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
  --seed-from-graph-field refiner_seed --diff-steps 800 --max-proposals 1139 \
  >"${RUN}/logs/tau800.out" 2>"${RUN}/logs/tau800.err"
PYTHONPATH="${CODE}/scripts:${CODE}/src" "${PY}" \
  "${CODE}/scripts/assemble_known1200_refined.py" \
  --accounting-jsonl "${RUN}/combined/all_requested_accounting.jsonl" \
  --refine-dir "${RUN}/tau800/refine" --output-dir "${RUN}/tau800/generation" \
  --steps 800

trap - ERR
touch "${RUN}/_SUCCESS"

#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ms-critic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-ms-critic-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${H1A2_CRITIC_FEASIBILITY_DATA:?export H1A2_CRITIC_FEASIBILITY_DATA}"
readonly PYTHON="${H1A2_MATTERSIM_PYTHON:?export H1A2_MATTERSIM_PYTHON}"
readonly MODEL="${H1A2_MATTERSIM_MODEL:?export H1A2_MATTERSIM_MODEL}"
readonly RUN_DIR="${ROOT}/runs/mattersim_critic_feasibility_${SLURM_JOB_ID}"

for path in "${DATA}/_SUCCESS" "${DATA}/critic_feasibility.extxyz" "${DATA}/metadata.jsonl" "${PYTHON}" "${MODEL}"; do
  [[ -e "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
model_id	MatterSim-v1.0.0-5M.pth
model_path	${MODEL}
structures	${DATA}/critic_feasibility.extxyz
structure_stage	model494_tau800_pre-CHGNet
primary_label	CHGNet_relaxed_energy_per_atom
mattersim_mode	single_point
batch_size	1
workers	4
chgnet_at_inference	false
official_at_inference	false
rerank	false
training	false
EOF

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${rank}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PYTHON}" "${ROOT}/scripts/evaluate_mattersim_critic_feasibility.py" \
      --input-extxyz "${DATA}/critic_feasibility.extxyz" \
      --metadata "${DATA}/metadata.jsonl" \
      --output-dir "${RUN_DIR}" --model-path "${MODEL}" --device cuda \
      --rank "${rank}" --world-size 4 \
      >"${RUN_DIR}/rank${rank}.out" 2>"${RUN_DIR}/rank${rank}.err" &
  eval "p${rank}=\$!"
done
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

"${PYTHON}" "${ROOT}/scripts/evaluate_mattersim_critic_feasibility.py" \
  --input-extxyz "${DATA}/critic_feasibility.extxyz" \
  --metadata "${DATA}/metadata.jsonl" --output-dir "${RUN_DIR}" \
  --world-size 4 --merge
[[ -f "${RUN_DIR}/_SUCCESS" ]] || exit 3
printf '%s\n' "${RUN_DIR}"

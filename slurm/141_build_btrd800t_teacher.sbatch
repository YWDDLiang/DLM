#!/usr/bin/env bash
#SBATCH --job-name=btrd800t-teacher
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=06:00:00
#SBATCH --output=btrd800t-teacher-%j.out
#SBATCH --error=btrd800t-teacher-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export root}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export code root}"
readonly SOURCE_RUN="${ROOT}/runs/btrd_tau200_teacher_39184"
readonly SUBSET="${BTRD_SUBSET_DIR:?export BTRD subset}"
readonly SOURCE_DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly OUTPUT_DATA="${ROOT}/data/btrd_tau800t_sft_v1_20260902"
readonly REFINER_SCRIPT="${CODE_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly BUILDER="${CODE_ROOT}/scripts/build_btrd_sft_data.py"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/btrd_tau800t_teacher_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly PROPOSAL_SHA=f7d17636c2f71b4335739f33b405df1557214787c174420269193e9d64144c13
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc="$1"; line="$2"; cmd="$3"; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$cmd" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ ! -e "${OUTPUT_DATA}" && -f "${SOURCE_RUN}/_SUCCESS" && -f "${SUBSET}/_SUCCESS" ]]
[[ "$(sha256sum "${SOURCE_RUN}/combined/proposal_graphs.pt"|awk '{print $1}')" == "${PROPOSAL_SHA}" ]]
[[ "$(sha256sum "${REFINER}"|awk '{print $1}')" == "${REFINER_SHA}" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no GPUs}"; [[ "${#gpu[@]}" -eq 6 ]]
cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	btrd800t_endpoint_teacher_v1
source_proposals	job39184_same6038_no_body_rerun
requested_endpoint_rows	6144
proposal_rows	6038
model494_steps	800
refiner_seed	103117_by_btrd_index
anchor_rows	2048_plus_missing
energy_or_hull_label	false
EOF
sha256sum "${SOURCE_RUN}/combined/proposal_graphs.pt" "${SOURCE_RUN}/combined/all_requested_accounting.jsonl" \
 "${SUBSET}/selected.jsonl" "${REFINER}" "${REFINER_SCRIPT}" "${BUILDER}" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
 "${TORCHRUN}" --standalone --master-port 29641 --nproc_per_node=6 "${REFINER_SCRIPT}" \
 --proposal-graphs "${SOURCE_RUN}/combined/proposal_graphs.pt" --checkpoint "${REFINER}" \
 --output-dir "${RUN}/tau800" --crysllmgen-dir "${CRYS}" --batch-size 1 \
 --seed 103117 --seed-by-sample-index --diff-steps 800 --max-proposals 6144 \
 >"${RUN}/logs/tau800.out" 2>"${RUN}/logs/tau800.err"
metrics="${RUN}/tau800/refinement_metrics.json"; [[ -f "${metrics}" ]]
refined="$(${PYTHON} -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"; [[ -f "${refined}" ]]
PYTHONPATH="${CODE_ROOT}/src" "${PYTHON}" "${BUILDER}" --subset-jsonl "${SUBSET}/selected.jsonl" \
 --refined-pt "${refined}" --source-data-dir "${SOURCE_DATA}" --output-dir "${OUTPUT_DATA}" \
 --teacher-steps 800 >"${RUN}/logs/build-data.out" 2>"${RUN}/logs/build-data.err"
[[ -f "${OUTPUT_DATA}/_SUCCESS" ]]
(cd "${OUTPUT_DATA}" && sha256sum -c SHA256SUMS)
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

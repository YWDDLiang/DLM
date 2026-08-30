#!/usr/bin/env bash
#SBATCH --job-name=h1a2-rich-refine-fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:5
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=h1a2-rich-refine-fix-%j.out
#SBATCH --error=h1a2-rich-refine-fix-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PARENT="${ROOT}/runs/rich_recovery_canary_generation_38363"
readonly RUN="${ROOT}/runs/rich_recovery_refine_completion_${SLURM_JOB_ID}"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" \
    >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly CONTRACT_SHA=9471cdd6ea370548b53738a23b6326be6f44049ea3f6f75534a31f87c56671c3
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly PARENT_CONTRACT_SHA=f35d62150d81b8698e17096a0a7538b3a809d83e9317d981ef94deedee32825d
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498
readonly S17_RCF_SHA=9ed8fab8b35742388cbad8e8831c21df6b4a6ece68924d614c19bc8539eb4695
readonly S17_R0_SHA=5dd212122860b9057e7779a4ca34fcad9d7f569b778ca671954308d29f255d52
readonly S18_M0_SHA=3a7db1ad44a1f4b95fbbadeb56421eb3a5b52cbad2a5ac1af8781498b393b4c6
readonly S18_RCF_SHA=914a883fe1add18de980bf012f46ec0d8883398239f5bdafd3628758b27c7d44
readonly S18_R0_SHA=6a3ba284b5b24c6150c6ce9f8d95789906e5f02b002845b6adbe449d6f55749e
readonly COMPLETED_S17_M0_SHA=1f25753e76e07a856922b04e8aab4641697dd4cb64ae0f11454e50de61c3b35b

mkdir -p "${ROOT}/runs/contracts"
{
  printf 'schema=h1a2_rich_recovery_refine_completion_v1\n'
  printf 'parent_job=38363\n'
  printf 'parent_contract=%s\n' "${PARENT_CONTRACT_SHA}"
  printf 's17_rcf=%s\n' "${S17_RCF_SHA}"
  printf 's17_r0=%s\n' "${S17_R0_SHA}"
  printf 's18_m0=%s\n' "${S18_M0_SHA}"
  printf 's18_rcf=%s\n' "${S18_RCF_SHA}"
  printf 's18_r0=%s\n' "${S18_R0_SHA}"
  printf 'completed_s17_m0=%s\n' "${COMPLETED_S17_M0_SHA}"
  printf 'model494=%s\n' "${REFINER_SHA}"
  printf 'refiner_script=%s\n' "${REFINER_SCRIPT_SHA}"
  printf 'refiner_seeds=101117,102117\n'
  printf 'tau=800\n'
  printf 'missing_cells=17RCF,17R0,18M0,18RCF,18R0\n'
  printf 'rerun_body=false\n'
} >"${RUN}/contract_payload.txt"
[[ "$(sha256sum "${RUN}/contract_payload.txt" | awk '{print $1}')" == "${CONTRACT_SHA}" ]] || exit 2
mkdir "${CONTRACT_DIR}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\th1a2_rich_recovery_refine_completion_v1\n'
  printf 'parent_job\t38363\n'
  printf 'completed_parent_cell\tstream17/M0\n'
  printf 'missing_cells\tstream17/RCF,stream17/R0,stream18/M0,stream18/RCF,stream18/R0\n'
  printf 'body_rerun\tfalse\n'
  printf 'refiner_tau\t800\n'
  printf 'science_parameters_changed\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

[[ -f "${PARENT}/_FAILED" ]] || exit 2
[[ "$(sha256sum "${PARENT}/stream17/M0/refine/dlm_refined_mp_256.pt" | awk '{print $1}')" == "${COMPLETED_S17_M0_SHA}" ]] || exit 2
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2

check_graph() {
  local stream="$1" arm="$2" expected="$3"
  local path="${PARENT}/stream${stream}/${arm}/body/proposal_graphs.pt"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
}
check_graph 17 RCF "${S17_RCF_SHA}"
check_graph 17 R0 "${S17_R0_SHA}"
check_graph 18 M0 "${S18_M0_SHA}"
check_graph 18 RCF "${S18_RCF_SHA}"
check_graph 18 R0 "${S18_R0_SHA}"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 5 ]] || exit 2

run_refine() {
  local slot="$1" stream="$2" arm="$3" seed="$4"
  local device output metrics output_file
  device="${GPU_DEVICES[${slot}]}"
  output="${RUN}/stream${stream}/${arm}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${PARENT}/stream${stream}/${arm}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed "${seed}" \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 256 \
    >"${RUN}/stream${stream}-${arm}.out" \
    2>"${RUN}/stream${stream}-${arm}.err"
  metrics="${output}/refinement_metrics.json"
  [[ -f "${metrics}" ]] || exit 3
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]] || exit 3
  touch "${output}/_SUCCESS"
}

run_refine 0 17 RCF 101117 & p0=$!
run_refine 1 17 R0 101117 & p1=$!
run_refine 2 18 M0 102117 & p2=$!
run_refine 3 18 RCF 102117 & p3=$!
run_refine 4 18 R0 102117 & p4=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"
wait "${p4}"

for cell in stream17/RCF stream17/R0 stream18/M0 stream18/RCF stream18/R0; do
  [[ -f "${RUN}/${cell}/_SUCCESS" ]] || exit 3
done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

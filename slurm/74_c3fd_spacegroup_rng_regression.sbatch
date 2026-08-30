#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sg-rng-regress
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=h1a2-sg-rng-regress-%j.out
#SBATCH --error=h1a2-sg-rng-regress-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly SOURCE="${ROOT}/runs/c3fd_v25_online_canary_36608"
readonly CHECKPOINT17="${SOURCE}/train_seed17/checkpoint.pt"
readonly CHECKPOINT18="${SOURCE}/train_seed18/checkpoint.pt"
readonly LEGACY17="${SOURCE}/canary_seed17"
readonly LEGACY18="${SOURCE}/canary_seed18"
readonly OUTPUT="${ROOT}/data/c3fd_sg_rng_regression_v1_20260830"
readonly PREPARING="${OUTPUT}.preparing"
readonly RUN="${ROOT}/runs/c3fd_sg_rng_regression_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CONTRACT_SHA=235314e236d8dd0d0270087c859363e2950dd472b01d61c16b25154d4824df01
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly SEED17_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly SEED18_SHA=d2e31996d034ba0f452ba72906539d6067bf3d7bf8969940c0f7961011b02b07
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly SAMPLER_SHA=0fca6923b8600f7d94c2af1c0409038f515d481f572ec55a4861691055c1a7f3
readonly COMPARATOR_SHA=7043512e516903b313db6bfc808a16e76ceaf84a59b390e58a3772a2fc701e94
readonly LEGACY17_PLANS_SHA=3be804682c373191c87eecac5d4fd71fa662e9edb71ae67ab7943bc5d5d60ef4
readonly LEGACY17_RAW_SHA=3f657548aeae460bf01c531ca38f192ddabbc2999795e096c6e6789bed00b514
readonly LEGACY18_PLANS_SHA=1b6c3ca545c8f0ed41c2133f3c88a2eacf1da57b90e561e7ca9bdd54935fe459
readonly LEGACY18_RAW_SHA=3d76368b22ee2f71169b1fa0e0f0f6108094ed0b739fa92694f39149689943e4

mkdir -p "${ROOT}/runs/contracts"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${PREPARING}" ]]; then
    printf '%s\n' "${SLURM_JOB_ID}" >"${PREPARING}/FAILED_JOB_ID"
    mv "${PREPARING}" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

mkdir "${CONTRACT_DIR}"
mkdir "${PREPARING}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\t%s\n' 'h1a2_c3fd_spacegroup_rng_regression_v1'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'samples_per_checkpoint\t32\n'
  printf 'seeds\t17,18\n'
  printf 'temperature\t0.9\n'
  printf 'top_p\t0.95\n'
  printf 'top_k\t0\n'
  printf 'reachability\tpauling_bitset\n'
  printf 'spacegroup_mode\tindependent_head\n'
  printf 'device\tcpu\n'
  printf 'cpus\t16\n'
  printf 'gpu_jobs_used\t0\n'
  printf 'outcomes_read\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ "$(sha256sum "${CHECKPOINT17}" | awk '{print $1}')" == "${SEED17_SHA}" ]] || exit 2
[[ "$(sha256sum "${CHECKPOINT18}" | awk '{print $1}')" == "${SEED18_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')" == "${VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/sample_c3fd_plans.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/audit_c3fd_spacegroup_rng_regression.py" | awk '{print $1}')" == "${COMPARATOR_SHA}" ]] || exit 2
[[ "$(sha256sum "${LEGACY17}/plans_for_dlm.jsonl" | awk '{print $1}')" == "${LEGACY17_PLANS_SHA}" ]] || exit 2
[[ "$(sha256sum "${LEGACY17}/raw_generations.jsonl" | awk '{print $1}')" == "${LEGACY17_RAW_SHA}" ]] || exit 2
[[ "$(sha256sum "${LEGACY18}/plans_for_dlm.jsonl" | awk '{print $1}')" == "${LEGACY18_PLANS_SHA}" ]] || exit 2
[[ "$(sha256sum "${LEGACY18}/raw_generations.jsonl" | awk '{print $1}')" == "${LEGACY18_RAW_SHA}" ]] || exit 2

sha256sum \
  "${CHECKPOINT17}" "${CHECKPOINT18}" "${DATA}/vocabulary.json" \
  "${ROOT}/scripts/sample_c3fd_plans.py" \
  "${ROOT}/scripts/audit_c3fd_spacegroup_rng_regression.py" \
  "${LEGACY17}/plans_for_dlm.jsonl" "${LEGACY17}/raw_generations.jsonl" \
  "${LEGACY18}/plans_for_dlm.jsonl" "${LEGACY18}/raw_generations.jsonl" \
  >"${RUN}/INPUTS.sha256"

run_corrected() {
  local seed="$1" checkpoint="$2"
  env \
    CUDA_VISIBLE_DEVICES= \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    "${PYTHON}" "${ROOT}/scripts/sample_c3fd_plans.py" \
    --checkpoint "${checkpoint}" \
    --data-dir "${DATA}" \
    --output-dir "${RUN}/corrected_seed${seed}" \
    --num-samples 32 \
    --start-index 0 \
    --seed "${seed}" \
    --temperature 0.9 \
    --top-p 0.95 \
    --top-k 0 \
    --pair-prior-weight 0 \
    --max-species 7 \
    --reachability-mode pauling_bitset \
    --spacegroup-mode independent_head \
    >"${RUN}/corrected_seed${seed}.out" \
    2>"${RUN}/corrected_seed${seed}.err"
}

run_corrected 17 "${CHECKPOINT17}" & pid17=$!
run_corrected 18 "${CHECKPOINT18}" & pid18=$!
wait "${pid17}"
wait "${pid18}"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${PYTHON}" "${ROOT}/scripts/audit_c3fd_spacegroup_rng_regression.py" \
  --pair "seed17=${LEGACY17},${RUN}/corrected_seed17" \
  --pair "seed18=${LEGACY18},${RUN}/corrected_seed18" \
  --output-dir "${OUTPUT}" \
  >"${RUN}/comparison.out" 2>"${RUN}/comparison.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
rmdir "${PREPARING}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"

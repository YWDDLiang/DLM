#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-token-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=c3fd-native-token-audit-%j.out
#SBATCH --error=c3fd-native-token-audit-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly SCRIPT="${ROOT}/scripts/audit_c3fd_native_teacher_sft.py"
readonly RUN="${ROOT}/runs/c3fd_native_teacher_token_audit_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SCRIPT_SHA=d0d666aabdf5d9841cad61d9f2a223681e7c9f0468feb14af8aed0343c97b67f
readonly CONFIG_SHA=5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a
readonly TOKENIZER_SHA=071eb20fe7bd601550b1b7838ff696d6c93b88f51257f753303d3df23163c381
readonly DATA_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly DATA_TRAIN_SHA=f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068
readonly DATA_VAL_SHA=bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ "$(sha256sum "${SCRIPT}" | awk '{print $1}')" == "${SCRIPT_SHA}" ]] || exit 2
[[ "$(sha256sum "${MODEL}/config.json" | awk '{print $1}')" == "${CONFIG_SHA}" ]] || exit 2
[[ "$(sha256sum "${MODEL}/tokenizer.json" | awk '{print $1}')" == "${TOKENIZER_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/manifest.json" | awk '{print $1}')" == "${DATA_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/train.jsonl" | awk '{print $1}')" == "${DATA_TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/val.jsonl" | awk '{print $1}')" == "${DATA_VAL_SHA}" ]] || exit 2

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${PYTHON}" "${SCRIPT}" \
  --model-path "${MODEL}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/audit" \
  --max-length 382 \
  >"${RUN}/audit.out" 2>"${RUN}/audit.err"

(cd "${RUN}/audit" && sha256sum -c SHA256SUMS)
"${PYTHON}" - "${RUN}/audit/TOKENIZATION_AUDIT.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['splits']['train']['rows']==27136
assert x['splits']['val']['rows']==9047
assert x['splits']['train']['over_limit']==0
assert x['splits']['val']['over_limit']==0
assert x['splits']['train']['prompt_answer_suffix_mismatches']==0
assert x['splits']['val']['prompt_answer_suffix_mismatches']==0
assert all(x['gate'].values()) and x['outcomes_read'] is False
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

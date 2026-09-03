#!/usr/bin/env bash
#SBATCH --job-name=potential-actions
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=potential-actions-%j.out
#SBATCH --error=potential-actions-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_spade_b14c457}"
readonly STATES="${ROOT}/data/potential_closure_states_v1_20260904"
readonly JOB39556="${ROOT}/runs/spad_energy_teacher_39556"
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly LLADA=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/sample_potential_closure_actions_${SLURM_JOB_ID}"
readonly OUTPUT="${ROOT}/data/potential_closure_actions_v1_20260904"

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 4 ]]
[[ ! -e "${OUTPUT}" ]]
[[ -f "${STATES}/_SUCCESS" ]]
[[ "$(wc -l <"${STATES}/states.jsonl")" -eq 2048 ]]
[[ -f "${JOB39556}/predictor_body/raw_generations.jsonl" ]]
[[ -f "${BS_POLICY}/adapter_config.json" ]]
[[ -f "${SOURCE}/src/scripts/sample_potential_closure_actions.py" ]]

cat >"${RUN}/contract.tsv" <<EOF
field	value
states	${STATES}/states.jsonl
state_source	job39556 train-only programs and native BS bodies
policy	${BS_POLICY}
groups	2048 in four balanced strata
distributed	4 A800, batch 8
temperature	0.7 fixed
proposal_attempts	at most 8 request-keyed attempts per group
retention	first distinct legal complete transactions in request order
energy_selection	false
official_or_test_outcomes	false
EOF

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master_port=29683 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/sample_potential_closure_actions.py" \
  --model-path "${LLADA}" --checkpoint-path "${BS_POLICY}" \
  --states "${STATES}/states.jsonl" --output-dir "${OUTPUT}" \
  --crysllmgen-dir "${CRYS}" --batch-size 8 --seed 20260904 \
  >"${RUN}/logs/sample.out" 2>"${RUN}/logs/sample.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/candidate_groups.jsonl")" -eq 2048 ]]
[[ -s "${OUTPUT}/proposal_attempts.jsonl" ]]
"${PYTHON}" - "${OUTPUT}/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["schema"] == "potential_closure_action_manifest_v1"
assert manifest["groups"] == 2048
assert manifest["groups_per_stratum"] == {
    "mp20_clean_cell": 512,
    "mp20_clean_site": 512,
    "on_policy_cell": 512,
    "on_policy_site": 512,
}
assert manifest["world_size"] == 4
assert manifest["batch_size"] == 8
assert manifest["maximum_proposal_attempts"] == 8
assert manifest["temperature"] == 0.7
assert manifest["energy_selection"] is False
assert manifest["dynamic_temperature"] is False
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

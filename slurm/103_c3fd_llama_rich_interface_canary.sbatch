#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=02:00:00
#SBATCH --output=c3fd-llama-canary-%j.out
#SBATCH --error=c3fd-llama-canary-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly DATA="${ROOT}/data/c3fd_llama_rich_expander_v1_20260831"
readonly TRAIN_RUN="${ROOT}/runs/c3fd_llama_rich_expanders_38933"
readonly SAMPLER="${ROOT}/src/scripts/sample_c3fd_llama_rich_expander.py"
readonly RUN="${ROOT}/runs/c3fd_llama_rich_interface_canary_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SAMPLER_SHA=7246c27d3ef28a11c8e1c259d8950e752cd4fdec4ddf1dbe207409eb0bfa8d07
readonly EXPANDER_SHA=a6a824118728c07ea6179613a0d295181ebb783471f80b0eb603f9a77da83007
readonly SAMPLE_SEED=86017
readonly REQUESTED=64

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${TRAIN_RUN}/_SUCCESS" ]]
[[ "$(sha256sum "${SAMPLER}" | awk '{print $1}')" == "${SAMPLER_SHA}" ]]
[[ "$(sha256sum "${ROOT}/src/crystal_dlm/c3fd_rich_expander.py" | awk '{print $1}')" == "${EXPANDER_SHA}" ]]

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_rich_interface_canary_v1
source	first64_MP20_validation_rows_fixed_order
cells	F84017,F84018,M84117,M84118
sample_seed	${SAMPLE_SEED}
temperature	0.9
top_p	0.95
top_k	50
formula_locked	true
outcomes_read	false
setting_selection	false
retry_replacement_rerank	false
EOF

sample_one() {
  local route="$1" seed="$2" gpu="$3"
  local checkpoint="${TRAIN_RUN}/train_${route}_seed${seed}/final"
  local output="${RUN}/${route}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SAMPLER}" \
    --model-path "${MODEL}" --checkpoint-path "${checkpoint}" \
    --condition-jsonl "${DATA}/${route}/val.jsonl" \
    --output-dir "${output}" --route "${route}" \
    --num-samples "${REQUESTED}" --seed "${SAMPLE_SEED}" \
    --max-new-tokens 64 --temperature 0.9 --top-p 0.95 --top-k 50 \
    >"${RUN}/logs/${route}_seed${seed}.out" \
    2>"${RUN}/logs/${route}_seed${seed}.err"
}

sample_one F 84017 "${allocated_gpus[0]}" & p0=$!
sample_one F 84018 "${allocated_gpus[1]}" & p1=$!
sample_one M 84117 "${allocated_gpus[2]}" & p2=$!
sample_one M 84118 "${allocated_gpus[3]}" & p3=$!
rc=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || rc=$?; done
[[ "${rc}" -eq 0 ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cells=[]
for route,seed in (("F",84017),("F",84018),("M",84117),("M",84118)):
    cell=root/f"{route}_seed{seed}"
    assert (cell/'_SUCCESS').is_file()
    x=json.load((cell/'sample_metrics.json').open())
    assert x['requested']==64 and x['decoded']+x['failed']>=x['requested']
    assert x['formula_changed']==0 and x['retry_replacement_rerank'] is False
    cells.append({"route":route,"training_seed":seed,"requested":64,"parsed":x['parsed'],"failed":x['failed']})
(root/'CANARY_FINAL.json').write_text(json.dumps({"schema":"c3fd_llama_rich_interface_canary_final_v1","cells":cells,"selection":False},indent=2,sort_keys=True)+'\n')
print(json.dumps(cells,sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

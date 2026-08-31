#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-plans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --output=c3fd-llama-plans-%j.out
#SBATCH --error=c3fd-llama-plans-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CANARY_RUN="${C3FD_LLAMA_CANARY_RUN:?export completed canary run}"
readonly FREEZE_RUN="${C3FD_LLAMA_FREEZE_RUN:?export completed prospective freeze run}"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly TRAIN_RUN="${ROOT}/runs/c3fd_llama_rich_expanders_38933"
readonly CONDITIONS="${ROOT}/data/c3fd_llama_prospective_conditions_seed21_256_v1_20260831"
readonly SAMPLER="${ROOT}/src/scripts/sample_c3fd_llama_rich_expander.py"
readonly RUN="${ROOT}/runs/c3fd_llama_prospective_plans_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SAMPLER_SHA=7246c27d3ef28a11c8e1c259d8950e752cd4fdec4ddf1dbe207409eb0bfa8d07
readonly SAMPLE_SEED=87017

mkdir "${RUN}"
mkdir "${RUN}/logs" "${RUN}/conditions"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${TRAIN_RUN}/_SUCCESS" && -f "${CANARY_RUN}/_SUCCESS" && -f "${FREEZE_RUN}/_SUCCESS" ]]
[[ -f "${CONDITIONS}/_SUCCESS" ]]
[[ "$(sha256sum "${SAMPLER}" | awk '{print $1}')" == "${SAMPLER_SHA}" ]]

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_prospective_plan_expansion_v1
routes	F,M
training_seed_assignment	indices0_127:seedA;indices128_255:seedB
F_seeds	84017,84018
M_seeds	84117,84118
sample_seed	${SAMPLE_SEED}
temperature	0.9
top_p	0.95
top_k	50
one_plan_per_request	true
retry_replacement_rerank	false
outcomes_read	false
EOF

"${PYTHON}" - "${CONDITIONS}" "${RUN}/conditions" <<'PY'
import json,pathlib,sys
source=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
for route in ('F','M'):
    rows=[json.loads(line) for line in (source/f'{route}.jsonl').open() if line.strip()]
    assert len(rows)==256 and [int(row['sample_idx']) for row in rows]==list(range(256))
    for label,subset in (('A',rows[:128]),('B',rows[128:])):
        with (output/f'{route}_{label}.jsonl').open('x') as handle:
            for row in subset: handle.write(json.dumps(row,sort_keys=True)+'\n')
PY

sample_one() {
  local route="$1" seed="$2" half="$3" gpu="$4"
  local checkpoint="${TRAIN_RUN}/train_${route}_seed${seed}/final"
  local output="${RUN}/${route}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SAMPLER}" \
    --model-path "${MODEL}" --checkpoint-path "${checkpoint}" \
    --condition-jsonl "${RUN}/conditions/${route}_${half}.jsonl" \
    --output-dir "${output}" --route "${route}" --num-samples 128 \
    --seed "${SAMPLE_SEED}" --max-new-tokens 64 \
    --temperature 0.9 --top-p 0.95 --top-k 50 \
    >"${RUN}/logs/${route}_seed${seed}.out" \
    2>"${RUN}/logs/${route}_seed${seed}.err"
}

sample_one F 84017 A "${allocated_gpus[0]}" & p0=$!
sample_one F 84018 B "${allocated_gpus[1]}" & p1=$!
sample_one M 84117 A "${allocated_gpus[2]}" & p2=$!
sample_one M 84118 B "${allocated_gpus[3]}" & p3=$!
rc=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || rc=$?; done
[[ "${rc}" -eq 0 ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); report={"schema":"c3fd_llama_prospective_plan_expansion_final_v1","routes":{},"selection":False}
for route,seeds in (("F",(84017,84018)),("M",(84117,84118))):
    raw=[]; plans=[]; cells=[]
    for seed in seeds:
        cell=root/f'{route}_seed{seed}'
        assert (cell/'_SUCCESS').is_file()
        metrics=json.load((cell/'sample_metrics.json').open())
        cells.append({"seed":seed,"requested":metrics['requested'],"parsed":metrics['parsed'],"failed":metrics['failed']})
        raw += [json.loads(line) for line in (cell/'raw_generations.jsonl').open() if line.strip()]
        plans += [json.loads(line) for line in (cell/'plans_for_dlm.jsonl').open() if line.strip()]
    raw.sort(key=lambda row:int(row['sample_idx'])); plans.sort(key=lambda row:int(row['sample_idx']))
    assert len(raw)==256 and [int(row['sample_idx']) for row in raw]==list(range(256))
    assert len({int(row['sample_idx']) for row in plans})==len(plans)
    out=root/f'route_{route}'; out.mkdir()
    for name,rows in (("raw_generations.jsonl",raw),("plans_for_dlm.jsonl",plans)):
        with (out/name).open('x') as handle:
            for row in rows: handle.write(json.dumps(row,sort_keys=True)+'\n')
    report['routes'][route]={"requested":256,"parsed":len(plans),"failed":256-len(plans),"training_seed_cells":cells}
(root/'PLAN_EXPANSION_FINAL.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors \
  ! -name soft_prefix_projector.pt -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=g2-full-ab-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=04:00:00
#SBATCH --output=g2-full-ab-raw-%j.out
#SBATCH --error=g2-full-ab-raw-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly A="${G2_FULL_A_POLICY:?export completed A step1696 checkpoint}"
readonly B="${G2_FULL_B_POLICY:?export completed B step1696 checkpoint}"
readonly A_POLICY_SHA="${G2_FULL_A_POLICY_SHA256:?pin A LoRA SHA256}"
readonly A_RELATION_SHA="${G2_FULL_A_RELATION_SHA256:?pin A relation SHA256}"
readonly A_CONFIG_SHA="${G2_FULL_A_CONFIG_SHA256:?pin A relation config SHA256}"
readonly B_POLICY_SHA="${G2_FULL_B_POLICY_SHA256:?pin B LoRA SHA256}"
readonly B_RELATION_SHA="${G2_FULL_B_RELATION_SHA256:?pin B relation SHA256}"
readonly B_CONFIG_SHA="${G2_FULL_B_CONFIG_SHA256:?pin B relation config SHA256}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly PLANS="${ROOT}/runs/c3fd_llama_g2_final_typed_plans_39128/plans/plans_for_dlm.jsonl"
readonly PLANS_SHA=5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb
readonly SAMPLER="${CODE_ROOT}/src/scripts/sample_sgtc_l6.py"
readonly SAMPLER_SHA="${G2_FULL_SAMPLER_SHA256:?pin sampler SHA256}"
readonly ASSEMBLER="${CODE_ROOT}/scripts/assemble_raw_body_repeat.py"
readonly ASSEMBLER_SHA="${G2_FULL_ASSEMBLER_SHA256:?pin assembler SHA256}"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly DIRECT_SHA="${G2_FULL_DIRECT_SHA256:?pin direct evaluator SHA256}"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly GT_SHA="${G2_FULL_GT_SHA256:?pin MP20 test SHA256}"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly BASE_REPORT="${ROOT}/runs/compact_v2_periodic_relation_g2_final_eval_39139/BASE/raw/evaluation/direct/report.json"
readonly BASE_REPORT_SHA="${G2_FULL_BASE_REPORT_SHA256:?pin matched BASE report SHA256}"
readonly RUN="${ROOT}/runs/g2_full_ab_raw_screen_${SLURM_JOB_ID}"
readonly BODY_PLANS="${RUN}/plans_for_body.jsonl"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
for spec in \
 "${PLANS_SHA}:${PLANS}" \
 "${A_POLICY_SHA}:${A}/adapter_model.safetensors" \
 "${A_RELATION_SHA}:${A}/periodic_relation_adapter.pt" \
 "${A_CONFIG_SHA}:${A}/periodic_relation_config.json" \
 "${B_POLICY_SHA}:${B}/adapter_model.safetensors" \
 "${B_RELATION_SHA}:${B}/periodic_relation_adapter.pt" \
 "${B_CONFIG_SHA}:${B}/periodic_relation_config.json" \
 "${SAMPLER_SHA}:${SAMPLER}" "${ASSEMBLER_SHA}:${ASSEMBLER}" \
 "${DIRECT_SHA}:${DIRECT}" "${GT_SHA}:${GT_CSV}" \
 "${BASE_REPORT_SHA}:${BASE_REPORT}"; do
 expected="${spec%%:*}"; path="${spec#*:}"
 [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
[[ "$(wc -l <"${PLANS}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	g2_full_ab_same_plan_raw_screen_v1
plans_sha256	${PLANS_SHA}
arms	A,B
training_seed_each	81017
stream	17
dlm_seed	91117
temperature	0.7
schedule	exact_axis
requested_per_arm	256
trajectory_per_plan	1
planner_resampled	false
retry_filter_replacement_rerank_best_of_n	false
technical_floor_body_comp	244
directional_direct_gate	greater_than_121
meaningful_direct_gate	at_least_134
B_promotion_gate	body_not_lower_and_direct_at_least_A_plus_8_or_later_raw_energy_CI_better
model494	false_in_raw_stage
official_query	false
claim_scope	post_outcome_matched_development
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${PLANS}" \
 "${A}/adapter_model.safetensors" "${A}/periodic_relation_adapter.pt" "${A}/periodic_relation_config.json" \
 "${B}/adapter_model.safetensors" "${B}/periodic_relation_adapter.pt" "${B}/periodic_relation_config.json" \
 "${SAMPLER}" "${ASSEMBLER}" "${DIRECT}" "${GT_CSV}" "${BASE_REPORT}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" - "${PLANS}" "${BODY_PLANS}" <<'PY'
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()/'src'))
from crystal_dlm.composition_identity import identity_from_plan_state,identity_text
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==256 and [int(row['sample_idx']) for row in rows]==list(range(256))
with open(sys.argv[2],'x') as handle:
 for row in rows:
  row=dict(row); row['reduced_composition_identity']=identity_text(identity_from_plan_state(row['plan_state']))
  handle.write(json.dumps(row,sort_keys=True)+'\n')
PY

run_body(){
 local method="$1" checkpoint="$2" device="$3" output="${RUN}/${1}"
 mkdir "${output}"
 env CUDA_VISIBLE_DEVICES="${device}" PYTHONPATH="${CODE_ROOT}/src" \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SAMPLER}" --model-path "${MODEL}" --checkpoint-path "${checkpoint}" \
  --periodic-relation-checkpoint "${checkpoint}" --periodic-relation-rank 64 \
  --prompt-jsonl "${BODY_PLANS}" --output-dir "${output}/body" --crysllmgen-dir "${CRYS}" \
  --num-samples 256 --temperature 0.7 --seed 91117 >"${output}/body.out" 2>"${output}/body.err"
 [[ -f "${output}/body/_SUCCESS" ]]
}
run_body A "${A}" "${allocated[0]}" & p0=$!
run_body B "${B}" "${allocated[1]}" & p1=$!
wait "${p0}"; wait "${p1}"

run_direct(){
 local method="$1" output="${RUN}/${1}"
 "${PYTHON}" "${ASSEMBLER}" --body-dir "${output}/body" \
  --output-dir "${output}/generation" --seed 17 --dlm-seed 91117 \
  --denominator 256 --source-arm full_axis --arm-label candidate
 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=8 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${output}/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${output}/direct" --eval-model-name mp20 >"${output}/direct.out" 2>"${output}/direct.err"
 [[ -f "${output}/direct/report.json" ]]
}
run_direct A & d0=$!
run_direct B & d1=$!
wait "${d0}"; wait "${d1}"

"${PYTHON}" - "${RUN}" "${BASE_REPORT}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); base=json.load(open(sys.argv[2]))
result={'schema':'g2_full_ab_same_plan_raw_final_v1','requested':256,
        'base':{'body':int(base['generation_succeeded']),'comp':int(base['comp_valid_count']),'direct':int(base['valid_count'])},'methods':{}}
for method in ('A','B'):
 report=json.load((root/method/'direct/report.json').open())
 assert int(report['attempts'])==256 and report['retry_or_replacement_used'] is False
 body=int(report['generation_succeeded']); comp=int(report['comp_valid_count']); direct=int(report['valid_count'])
 result['methods'][method]={'body':body,'comp':comp,'direct':direct,
   'technical_pass':body>=244 and comp>=244,'directional_pass':direct>121,'meaningful_pass':direct>=134}
a=result['methods']['A']; b=result['methods']['B']
result['B_direct_promotion_pass']=b['body']>=a['body'] and b['direct']>=a['direct']+8
(root/'G2_FULL_AB_RAW_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"

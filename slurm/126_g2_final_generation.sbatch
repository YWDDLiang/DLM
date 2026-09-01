#!/usr/bin/env bash
#SBATCH --job-name=g2-final-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=05:00:00
#SBATCH --output=g2-final-gen-%j.out
#SBATCH --error=g2-final-gen-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly BASE="${ROOT}/runs/c3fd_native_teacher_sft_38703/train_seed82017/checkpoints/step-3392"
readonly G2="${ROOT}/runs/compact_v2_periodic_relation_g2_39107/train/checkpoints/step-348"
readonly PLANS="${ROOT}/runs/c3fd_llama_g2_final_typed_plans_39128/plans/plans_for_dlm.jsonl"
readonly SAMPLER="${CODE_ROOT}/src/scripts/sample_sgtc_l6.py"
readonly REFINER_SCRIPT="${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_final_generation_${SLURM_JOB_ID}"
readonly BODY_PLANS="${RUN}/plans_for_body.jsonl"

readonly PLANS_SHA=5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb
readonly BASE_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20
readonly G2_SHA=153998f0fe9196cbd4ad879068dc609d9ccd674234582bad7315a4bccb77bff4
readonly G2_RELATION_SHA=c0c057a29580476ddb9854c1bb8e90027a353291a493f43ad8ca0aaf0c7969cd
readonly G2_RELATION_CONFIG_SHA=0db88e018554c45c986271c1537dc19d7b13f4c6d4e16f46c37453515f94972c
readonly SAMPLER_SHA=b1749f1701af67e70578bd73a169bbfa4902c0416cee41d01d22193e8d29fb92
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
actual_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)"; [[ "${actual_commit}" == "${CODE_COMMIT}" ]]
for spec in "${PLANS_SHA}:${PLANS}" "${BASE_SHA}:${BASE}/adapter_model.safetensors" \
 "${G2_SHA}:${G2}/adapter_model.safetensors" "${G2_RELATION_SHA}:${G2}/periodic_relation_adapter.pt" \
 "${G2_RELATION_CONFIG_SHA}:${G2}/periodic_relation_config.json" "${SAMPLER_SHA}:${SAMPLER}" \
 "${REFINER_SCRIPT_SHA}:${REFINER_SCRIPT}" "${REFINER_SHA}:${REFINER}"; do expected="${spec%%:*}"; path="${spec#*:}"; [[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]]; done
[[ -f "$(dirname "${PLANS}")/_SUCCESS" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no GPUs}"; [[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_final_generation_v1
plans_sha256	${PLANS_SHA}
arms	BASE,G2
stream	17
dlm_seed	91117
refiner_seed	101117
temperature	0.7
schedule	exact_axis
model494_tau	800
requested_per_arm	256
trajectory_per_plan	1
retry_rerank_replacement_best_of_n	false
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${PLANS}" "${BASE}/adapter_model.safetensors" \
 "${G2}/adapter_model.safetensors" "${G2}/periodic_relation_adapter.pt" \
 "${G2}/periodic_relation_config.json" "${SAMPLER}" "${REFINER_SCRIPT}" "${REFINER}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" - "${PLANS}" "${BODY_PLANS}" <<'PY'
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()/'src'))
from crystal_dlm.composition_identity import identity_from_plan_state,identity_text
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; assert len(rows)==256
with open(sys.argv[2],'x') as h:
 for row in rows:
  row=dict(row); row['reduced_composition_identity']=identity_text(identity_from_plan_state(row['plan_state']))
  h.write(json.dumps(row,sort_keys=True)+'\n')
PY

run_arm(){
 local arm="$1" checkpoint="$2" device="$3" relation="$4" output="${RUN}/arm_${1}"
 mkdir "${output}"
 relation_args=()
 if [[ "${relation}" == true ]]; then relation_args=(--periodic-relation-checkpoint "${checkpoint}" --periodic-relation-rank 64); fi
 env CUDA_VISIBLE_DEVICES="${device}" PYTHONPATH="${CODE_ROOT}/src" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SAMPLER}" --model-path "${MODEL}" --checkpoint-path "${checkpoint}" "${relation_args[@]}" \
  --prompt-jsonl "${BODY_PLANS}" --output-dir "${output}/body" --crysllmgen-dir "${CRYS}" \
  --num-samples 256 --temperature 0.7 --seed 91117 >"${output}/body.out" 2>"${output}/body.err"
 [[ -f "${output}/body/_SUCCESS" ]]
 env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${REFINER_SCRIPT}" --proposal-graphs "${output}/body/proposal_graphs.pt" --checkpoint "${REFINER}" \
  --output-dir "${output}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 --seed 101117 \
  --seed-by-sample-index --diff-steps 800 --max-proposals 256 >"${output}/refine.out" 2>"${output}/refine.err"
 metrics="${output}/refine/refinement_metrics.json"; [[ -f "${metrics}" ]]
 path="$(${PYTHON} -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"; [[ -f "${path}" ]]
 touch "${output}/_SUCCESS"
}
run_arm BASE "${BASE}" "${allocated[0]}" false & p0=$!
run_arm G2 "${G2}" "${allocated[1]}" true & p1=$!
wait "${p0}"; wait "${p1}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=g2-known1159
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=520G
#SBATCH --time=05:00:00
#SBATCH --output=g2-known1159-%j.out
#SBATCH --error=g2-known1159-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:-unrecorded}"
readonly SPLITS="${KNOWN1200_SPLITS:?export frozen main1000/remainder cohort}"
readonly SPLIT_MANIFEST_SHA="${KNOWN1200_SPLIT_MANIFEST_SHA256:?pin split manifest}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${KNOWN1200_POLICY:-${ROOT}/runs/g2_full_epoch_A_39164/train/checkpoints/step-1696}"
readonly METHOD_LABEL="${KNOWN1200_METHOD_LABEL:-G2-PBC-R_step1696}"
readonly RUN_PREFIX="${KNOWN1200_RUN_PREFIX:-g2_known1200_baseline_fast}"
readonly SAMPLER="${CODE_ROOT}/src/scripts/sample_sgtc_l6.py"
readonly SHARDER="${CODE_ROOT}/scripts/shard_plan_jsonl.py"
readonly ASSEMBLER="${CODE_ROOT}/scripts/assemble_raw_body_repeat.py"
readonly FAST="${CODE_ROOT}/scripts/run_direct_validity_fast.py"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly SNAPSHOT="${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly RUN="${ROOT}/runs/${RUN_PREFIX}_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly POLICY_SHA="${KNOWN1200_POLICY_SHA256:-712c941922611ea3640b171319c8ae9a4cbdbf31f83cdba6f61046b9de61bfce}"
readonly RELATION_SHA="${KNOWN1200_RELATION_SHA256:-675d68b9e9c0aeb7f3383050b8c85716aa2f7c9303d5a8e1e75b0761db54b9f8}"
readonly RELATION_CONFIG_SHA="${KNOWN1200_RELATION_CONFIG_SHA256:-7e3431e75831f11fd48f564068617f3d3f6fa1a6dc89754aa72cd9c463bee353}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
actual_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
printf 'field\texpected\tobserved\ncode_commit\t%s\t%s\n' "${CODE_COMMIT}" "${actual_commit}" >"${RUN}/runtime_audit.tsv"
for path in "${SAMPLER}" "${SHARDER}" "${ASSEMBLER}" "${FAST}"; do [[ -f "${path}" ]]; done
for spec in "${SPLIT_MANIFEST_SHA}:${SPLITS}/manifest.json" "${POLICY_SHA}:${POLICY}/adapter_model.safetensors" \
 "${RELATION_SHA}:${POLICY}/periodic_relation_adapter.pt" "${RELATION_CONFIG_SHA}:${POLICY}/periodic_relation_config.json"; do
 expected="${spec%%:*}"; path="${spec#*:}"; [[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]]
done
[[ -f "${SPLITS}/_SUCCESS" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"; [[ "${#gpu[@]}" -eq 6 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	known1200_body_fast_direct_v2
plans	main1000_plus_remainder159
policy	${METHOD_LABEL}
stream	17
dlm_seed	91117
temperature	0.7
schedule	exact_axis
trajectory_per_plan	1
evaluation	body_plus_fast_Direct_only
model494_or_CHGNet_or_NU	false
retry_replacement_rerank_best_of_n	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${SPLITS}/manifest.json" \
 "${SPLITS}/main1000/plans_for_dlm.jsonl" "${SPLITS}/remainder/plans_for_dlm.jsonl" \
 "${POLICY}/adapter_model.safetensors" "${POLICY}/periodic_relation_adapter.pt" \
 "${POLICY}/periodic_relation_config.json" "${SAMPLER}" "${SHARDER}" "${ASSEMBLER}" "${FAST}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" "${SHARDER}" --input-jsonl "${SPLITS}/main1000/plans_for_dlm.jsonl" --output-dir "${RUN}/main_shards" --shard-size 200
"${PYTHON}" "${SHARDER}" --input-jsonl "${SPLITS}/remainder/plans_for_dlm.jsonl" --output-dir "${RUN}/remainder_shards" --shard-size 200

run_shard(){
 local slot="$1" name="$2" plans="$3" count="$4"
 local device="${gpu[$slot]}" out="${RUN}/${name}"
 mkdir "${out}"
 env CUDA_VISIBLE_DEVICES="${device}" PYTHONPATH="${CODE_ROOT}/src" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SAMPLER}" --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
  --periodic-relation-checkpoint "${POLICY}" --periodic-relation-rank 64 --prompt-jsonl "${plans}" \
  --output-dir "${out}/body" --crysllmgen-dir "${CRYS}" --num-samples "${count}" --expected-denominator "${count}" --temperature 0.7 --seed 91117 \
  >"${RUN}/logs/${name}-body.out" 2>"${RUN}/logs/${name}-body.err"
 [[ -f "${out}/body/_SUCCESS" ]]
 "${PYTHON}" "${ASSEMBLER}" --body-dir "${out}/body" --output-dir "${out}/generation" --seed 17 --dlm-seed 91117 \
  --denominator "${count}" --source-arm full_axis --arm-label control
 env SLURM_JOB_ID="${SLURM_JOB_ID}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${FAST}" --generation-jsonl "${out}/generation/generation.jsonl" --snapshot-root "${SNAPSHOT}" \
  --output-dir "${out}/fast_direct" --expected-denominator "${count}" >"${RUN}/logs/${name}-fast.out" 2>"${RUN}/logs/${name}-fast.err"
 [[ -f "${out}/fast_direct/_SUCCESS" ]]
}

pids=()
for index in 0 1 2 3 4; do
 run_shard "${index}" "main${index}" "${RUN}/main_shards/shard-$(printf '%03d' "${index}").jsonl" 200 & pids+=("$!")
done
run_shard 5 remainder "${RUN}/remainder_shards/shard-000.jsonl" 159 & pids+=("$!")
for pid in "${pids[@]}"; do wait "${pid}"; done

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cells=[]
for name in [f'main{i}' for i in range(5)]+['remainder']:
 report=json.load(open(root/name/'fast_direct/report.json')); cells.append({'name':name,**report})
result={
 'schema':'known1200_body_fast_direct_final_v2','requested':sum(x['attempts'] for x in cells),
 'generation_succeeded':sum(x['generation_succeeded'] for x in cells),
 'composition_valid':sum(x['comp_valid_count'] for x in cells),
 'structure_valid':sum(x['struct_valid_count'] for x in cells),
 'direct_joint':sum(x['valid_count'] for x in cells),'cells':cells,
 'expensive_direct_metrics_run':False,'retry_or_replacement_used':False,
}
assert result['requested']==1159
(root/'BASELINE_FAST_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

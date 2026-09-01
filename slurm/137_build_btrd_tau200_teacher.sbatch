#!/usr/bin/env bash
#SBATCH --job-name=btrd-teacher
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=10:00:00
#SBATCH --output=btrd-teacher-%j.out
#SBATCH --error=btrd-teacher-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly SUBSET="${BTRD_SUBSET_DIR:?export frozen BTRD subset}"
readonly SUBSET_MANIFEST_SHA="${BTRD_SUBSET_MANIFEST_SHA256:?pin subset manifest}"
readonly SOURCE_DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly OUTPUT_DATA="${ROOT}/data/btrd_tau200_sft_v1_20260901"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/g2_full_epoch_A_39164/train/checkpoints/step-1696"
readonly SAMPLER="${CODE_ROOT}/src/scripts/sample_sgtc_l6.py"
readonly MATERIALIZE="${CODE_ROOT}/scripts/materialize_btrd_execution_plans.py"
readonly SHARDER="${CODE_ROOT}/scripts/shard_plan_jsonl.py"
readonly COMBINE="${CODE_ROOT}/scripts/combine_btrd_proposal_graphs.py"
readonly REFINER_SCRIPT="${CODE_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly BUILDER="${CODE_ROOT}/scripts/build_btrd_sft_data.py"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/btrd_tau200_teacher_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly POLICY_SHA=712c941922611ea3640b171319c8ae9a4cbdbf31f83cdba6f61046b9de61bfce
readonly RELATION_SHA=675d68b9e9c0aeb7f3383050b8c85716aa2f7c9303d5a8e1e75b0761db54b9f8
readonly RELATION_CONFIG_SHA=7e3431e75831f11fd48f564068617f3d3f6fa1a6dc89754aa72cd9c463bee353
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA="${BTRD_SAMPLER_SHA256:?pin sampler}"
readonly MATERIALIZE_SHA="${BTRD_MATERIALIZE_SHA256:?pin materializer}"
readonly SHARDER_SHA="${BTRD_SHARDER_SHA256:?pin sharder}"
readonly COMBINE_SHA="${BTRD_COMBINE_SHA256:?pin combiner}"
readonly REFINER_SCRIPT_SHA="${BTRD_REFINER_SCRIPT_SHA256:?pin refiner wrapper}"
readonly BUILDER_SHA="${BTRD_BUILDER_SHA256:?pin data builder}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
[[ ! -e "${OUTPUT_DATA}" && -f "${SUBSET}/_SUCCESS" ]]
for spec in "${SUBSET_MANIFEST_SHA}:${SUBSET}/manifest.json" "${POLICY_SHA}:${POLICY}/adapter_model.safetensors" \
 "${RELATION_SHA}:${POLICY}/periodic_relation_adapter.pt" "${RELATION_CONFIG_SHA}:${POLICY}/periodic_relation_config.json" \
 "${REFINER_SHA}:${REFINER}" "${SAMPLER_SHA}:${SAMPLER}" "${MATERIALIZE_SHA}:${MATERIALIZE}" \
 "${SHARDER_SHA}:${SHARDER}" "${COMBINE_SHA}:${COMBINE}" "${REFINER_SCRIPT_SHA}:${REFINER_SCRIPT}" "${BUILDER_SHA}:${BUILDER}"; do
 expected="${spec%%:*}"; path="${spec#*:}"; [[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]]
done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"; [[ "${#gpu[@]}" -eq 6 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	btrd_tau200_teacher_v1
subset	8192_MP20_train_eval_disjoint
tau200_rows	6144
anchor_rows	2048
policy	G2-PBC-R_step1696
dlm_seed	91117
temperature	0.7
model494_steps	200
refiner_seed	103117_by_btrd_index
teacher_failure_policy	fallback_original_MP20_anchor_no_row_deletion
energy_label_or_selection	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${SUBSET}/manifest.json" "${SUBSET}/selected.jsonl" \
 "${SOURCE_DATA}/val.jsonl" "${POLICY}/adapter_model.safetensors" "${POLICY}/periodic_relation_adapter.pt" \
 "${POLICY}/periodic_relation_config.json" "${REFINER}" "${SAMPLER}" "${MATERIALIZE}" "${SHARDER}" \
 "${COMBINE}" "${REFINER_SCRIPT}" "${BUILDER}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" "${MATERIALIZE}" --subset-jsonl "${SUBSET}/selected.jsonl" --output-dir "${RUN}/teacher_plans" --expected-teacher-rows 6144
"${PYTHON}" "${SHARDER}" --input-jsonl "${RUN}/teacher_plans/plans_for_dlm.jsonl" --output-dir "${RUN}/plan_shards" --shard-size 1024

run_body(){
 local slot="$1" name="shard${1}" plans="${RUN}/plan_shards/shard-$(printf '%03d' "$1").jsonl" out="${RUN}/${name}"
 mkdir "${out}"
 env CUDA_VISIBLE_DEVICES="${gpu[$1]}" PYTHONPATH="${CODE_ROOT}/src" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SAMPLER}" --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
  --periodic-relation-checkpoint "${POLICY}" --periodic-relation-rank 64 --prompt-jsonl "${plans}" \
  --output-dir "${out}/body" --crysllmgen-dir "${CRYS}" --num-samples 1024 --temperature 0.7 --seed 91117 \
  >"${RUN}/logs/${name}-body.out" 2>"${RUN}/logs/${name}-body.err"
 [[ -f "${out}/body/_SUCCESS" ]]
}
pids=(); for index in 0 1 2 3 4 5; do run_body "${index}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "${pid}"; done

combine_args=(); for index in 0 1 2 3 4 5; do combine_args+=(--body-dir "${RUN}/shard${index}/body"); done
"${PYTHON}" "${COMBINE}" "${combine_args[@]}" --output-dir "${RUN}/combined" --expected-requested 6144

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
 "${TORCHRUN}" --standalone --master-port 29637 --nproc_per_node=6 "${REFINER_SCRIPT}" \
 --proposal-graphs "${RUN}/combined/proposal_graphs.pt" --checkpoint "${REFINER}" --output-dir "${RUN}/tau200" \
 --crysllmgen-dir "${CRYS}" --batch-size 1 --seed 103117 --seed-by-sample-index --diff-steps 200 --max-proposals 6144 \
 >"${RUN}/logs/tau200.out" 2>"${RUN}/logs/tau200.err"
metrics="${RUN}/tau200/refinement_metrics.json"; [[ -f "${metrics}" ]]
refined="$(${PYTHON} -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"; [[ -f "${refined}" ]]

PYTHONPATH="${CODE_ROOT}/src" "${PYTHON}" "${BUILDER}" --subset-jsonl "${SUBSET}/selected.jsonl" --refined-pt "${refined}" \
 --source-data-dir "${SOURCE_DATA}" --output-dir "${OUTPUT_DATA}" >"${RUN}/logs/build-data.out" 2>"${RUN}/logs/build-data.err"
[[ -f "${OUTPUT_DATA}/_SUCCESS" ]]
(cd "${OUTPUT_DATA}" && sha256sum -c SHA256SUMS)
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

#!/usr/bin/env bash
#SBATCH --job-name=g2-pstr256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=04:00:00
#SBATCH --output=g2-pstr256-%j.out
#SBATCH --error=g2-pstr256-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly BASE="${PSTR_BASE_POLICY:?export promoted G2-PBC-R step1696}"
readonly MASTER_PORT="${PSTR_MASTER_PORT:?set unique torchrun port}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly TRAINER="${CODE_ROOT}/src/scripts/llada_sft.py"
readonly OBJECTIVE="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_objective.py"
readonly OPS="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_ops.py"
readonly ADAPTER="${CODE_ROOT}/src/crystal_dlm/periodic_relation_adapter.py"
readonly RUNTIME="${CODE_ROOT}/src/crystal_dlm/periodic_relation_runtime.py"
readonly RUN="${ROOT}/runs/g2_pstr256_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly DATA_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly DATA_TRAIN_SHA=f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068
readonly DATA_VAL_SHA=bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1
readonly BASE_ADAPTER_SHA=712c941922611ea3640b171319c8ae9a4cbdbf31f83cdba6f61046b9de61bfce
readonly BASE_RELATION_SHA=675d68b9e9c0aeb7f3383050b8c85716aa2f7c9303d5a8e1e75b0761db54b9f8
readonly BASE_RELATION_CONFIG_SHA=7e3431e75831f11fd48f564068617f3d3f6fa1a6dc89754aa72cd9c463bee353
readonly TRAINER_SHA="${PSTR_TRAINER_SHA256:?pin trainer SHA256}"
readonly OBJECTIVE_SHA="${PSTR_OBJECTIVE_SHA256:?pin objective SHA256}"
readonly OPS_SHA="${PSTR_OPS_SHA256:?pin PBC ops SHA256}"
readonly ADAPTER_SHA="${PSTR_ADAPTER_SHA256:?pin adapter SHA256}"
readonly RUNTIME_SHA="${PSTR_RUNTIME_SHA256:?pin runtime SHA256}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
for spec in \
 "${DATA_MANIFEST_SHA}:${DATA}/manifest.json" "${DATA_TRAIN_SHA}:${DATA}/train.jsonl" "${DATA_VAL_SHA}:${DATA}/val.jsonl" \
 "${BASE_ADAPTER_SHA}:${BASE}/adapter_model.safetensors" "${BASE_RELATION_SHA}:${BASE}/periodic_relation_adapter.pt" \
 "${BASE_RELATION_CONFIG_SHA}:${BASE}/periodic_relation_config.json" "${TRAINER_SHA}:${TRAINER}" \
 "${OBJECTIVE_SHA}:${OBJECTIVE}" "${OPS_SHA}:${OPS}" "${ADAPTER_SHA}:${ADAPTER}" "${RUNTIME_SHA}:${RUNTIME}"; do
 expected="${spec%%:*}"; path="${spec#*:}"; [[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"; [[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	g2_periodic_sitewise_tail_risk_v1
initialization	${BASE}
training_seed	81017
optimizer_steps	256
world_size	2
effective_batch	16
learning_rate	1e-6_cosine_warmup10
trainable	periodic_relation_residual_only
tail_temperature	0.10
tail_mix	0.50
loss_weights	metric0.1_pairRDF0.1_overlap0.2_coordination0.05
PBC	strict_triclinic_125_image
species_margin	clamp_0.55_radius_sum_0.60A_1.40A
uncertainty_gate	false
planner_or_inference_change	false
checkpoint_selection	none_step256_only
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${DATA}/manifest.json" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
 "${BASE}/adapter_model.safetensors" "${BASE}/periodic_relation_adapter.pt" "${BASE}/periodic_relation_config.json" \
 "${TRAINER}" "${OBJECTIVE}" "${OPS}" "${ADAPTER}" "${RUNTIME}" >"${RUN}/INPUTS.sha256"

torchrun --standalone --master-port "${MASTER_PORT}" --nproc_per_node=2 "${TRAINER}" \
 --model-path "${MODEL}" --checkpoint-path "${BASE}" --periodic-relation-checkpoint "${BASE}" \
 --data-dir "${DATA}" --output-dir "${RUN}/train" --representation dynamic_v1 --dynamic-geometry-only \
 --max-length 382 --answer-token-count 87 --epochs 1 --max-train-steps 256 --batch-size 1 --grad-accum 8 \
 --lr 1e-6 --lr-scheduler cosine --warmup-steps 10 --min-lr-ratio 0.1 --save-steps 256 --skip-final-alias \
 --eval-steps 256 --eval-max-batches 50 --logging-steps 10 --dataloader-num-workers 4 --fail-on-truncation \
 --seed 81017 --data-seed 20260901 --no-lora --counterfactual-grounding-weight 0.0 \
 --periodic-metric-weight 0.1 --periodic-pair-rdf-weight 0.1 --periodic-overlap-weight 0.2 \
 --periodic-coordination-weight 0.05 --periodic-exact-triclinic-pbc --periodic-image-radius 2 \
 --periodic-species-margin-scale 0.55 --periodic-species-margin-floor 0.60 --periodic-species-margin-ceiling 1.40 \
 --periodic-overlap-tail-temperature 0.10 --periodic-overlap-tail-mix 0.50 --periodic-relation-rank 64 \
 --periodic-relation-circular-min-resultant 0.25 --periodic-relation-uncertainty-floor 0.25 \
 >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUT="${RUN}/train" CHECKPOINT="${RUN}/train/checkpoints/step-256"
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" && -f "${CHECKPOINT}/periodic_relation_config.json" ]]
[[ ! -e "${OUT}/final" ]]
mapfile -t checkpoints < <(find "${OUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -printf '%f\n')
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-256 ]]
"${PYTHON}" - "${OUT}" <<'PY'
import json,math,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(root/'run_config.json'))
assert c['use_lora'] is False and c['max_train_steps']==256 and c['seed']==81017
assert c['periodic_exact_triclinic_pbc'] is True and c['periodic_image_radius']==2
assert c['periodic_overlap_tail_temperature']==0.10 and c['periodic_overlap_tail_mix']==0.50
assert c['periodic_relation_uncertainty_gate'] is False
events=[json.loads(x) for x in open(root/'training_log.jsonl') if x.strip()]
train=[x for x in events if x.get('event')=='train']; assert train
assert any(x.get('event')=='eval' and int(x['step'])==256 for x in events)
for row in train:
 for key in ('loss','task_loss','grad_norm','periodic_relation_grad_norm','periodic_geometry_overlap_loss'):
  assert math.isfinite(float(row[key])),(key,row[key])
PY
sha256sum "${CHECKPOINT}/periodic_relation_adapter.pt" "${CHECKPOINT}/periodic_relation_config.json" >"${RUN}/step256_policy.sha256"
printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors ! -name periodic_relation_adapter.pt -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

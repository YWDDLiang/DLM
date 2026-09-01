#!/usr/bin/env bash
#SBATCH --job-name=g2-full-epoch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=08:00:00
#SBATCH --output=g2-full-epoch-%j.out
#SBATCH --error=g2-full-epoch-%j.err

set -Eeuo pipefail

readonly METHOD="${G2_FULL_METHOD:?set A or B}"
readonly MASTER_PORT="${G2_FULL_MASTER_PORT:?set a unique torchrun port}"
[[ "${METHOD}" == A || "${METHOD}" == B ]]

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly BASE="${ROOT}/runs/c3fd_native_teacher_sft_38703/train_seed82017/checkpoints/step-3392"
readonly AUDIT="${G2_FULL_GEOMETRY_AUDIT:?export passed immutable audit root}"
readonly AUDIT_SHA="${G2_FULL_GEOMETRY_AUDIT_SHA256:?pin audit report SHA256}"
readonly TRAINER="${CODE_ROOT}/src/scripts/llada_sft.py"
readonly OBJECTIVE="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_objective.py"
readonly OPS="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_ops.py"
readonly ADAPTER="${CODE_ROOT}/src/crystal_dlm/periodic_relation_adapter.py"
readonly RUNTIME="${CODE_ROOT}/src/crystal_dlm/periodic_relation_runtime.py"
readonly TRAINER_SHA="${G2_FULL_TRAINER_SHA256:?pin trainer SHA256}"
readonly OBJECTIVE_SHA="${G2_FULL_OBJECTIVE_SHA256:?pin objective SHA256}"
readonly OPS_SHA="${G2_FULL_OPS_SHA256:?pin shared geometry SHA256}"
readonly ADAPTER_SHA="${G2_FULL_ADAPTER_SHA256:?pin adapter SHA256}"
readonly RUNTIME_SHA="${G2_FULL_RUNTIME_SHA256:?pin runtime SHA256}"
readonly RUN="${ROOT}/runs/g2_full_epoch_${METHOD}_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

readonly MODEL_TREE_SHA=812a0ef72b2b2290b19b8beac32fd8ba0f4d32f308268f0c7914c4efd8755037
readonly DATA_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly DATA_TRAIN_SHA=f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068
readonly DATA_VAL_SHA=bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1
readonly BASE_ADAPTER_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20

export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

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

[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
[[ -f "${AUDIT}/_SUCCESS" && ! -e "${AUDIT}/_FAILED" ]]
[[ "$(sha256sum "${AUDIT}/GEOMETRY_CONTRACT_AUDIT.json" | awk '{print $1}')" == "${AUDIT_SHA}" ]]
for spec in \
  "${DATA_MANIFEST_SHA}:${DATA}/manifest.json" \
  "${DATA_TRAIN_SHA}:${DATA}/train.jsonl" \
  "${DATA_VAL_SHA}:${DATA}/val.jsonl" \
  "${BASE_ADAPTER_SHA}:${BASE}/adapter_model.safetensors" \
  "${TRAINER_SHA}:${TRAINER}" \
  "${OBJECTIVE_SHA}:${OBJECTIVE}" \
  "${OPS_SHA}:${OPS}" \
  "${ADAPTER_SHA}:${ADAPTER}" \
  "${RUNTIME_SHA}:${RUNTIME}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
actual_model_tree_sha="$(find "${MODEL}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${actual_model_tree_sha}" == "${MODEL_TREE_SHA}" ]]
IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#allocated_gpus[@]}" -eq 2 ]]

readonly GATE="$([[ "${METHOD}" == B ]] && printf true || printf false)"
cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	g2_full_epoch_two_method_v1
method	${METHOD}
initialization	${BASE}
base_adapter_sha256	${BASE_ADAPTER_SHA}
code_commit	${CODE_COMMIT}
training_seed	81017
data_seed	20260901
optimizer_steps	1696
source_epochs	1.0
world_size	2
effective_batch	16
learning_rate	5e-6_cosine_warmup25_min0.1
mask_policy	dynamic_geometry_only
loss_weights	metric0.1_pairRDF0.1_overlap0.2_coordination0.05
minimum_image	bounded_triclinic_27_image_shared_op
species_margin	clamp_0.55_radius_sum_0.60A_1.40A_normalized
relation_adapter	rank64_two_layer_zero_Wout
uncertainty_gate	${GATE}
uncertainty_gate_formula	detached_entropy_variance_floor0.25
checkpoint_selection	none_step1696_only
planner_resampled	false
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${AUDIT}/GEOMETRY_CONTRACT_AUDIT.json" \
  "${DATA}/manifest.json" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${BASE}/adapter_model.safetensors" "${TRAINER}" "${OBJECTIVE}" "${OPS}" \
  "${ADAPTER}" "${RUNTIME}" >"${RUN}/INPUTS.sha256"

gate_args=(--periodic-relation-uncertainty-floor 0.25)
if [[ "${METHOD}" == B ]]; then
  gate_args+=(--periodic-relation-uncertainty-gate)
fi

torchrun --standalone --master-port "${MASTER_PORT}" --nproc_per_node=2 "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 1696 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 25 \
  --min-lr-ratio 0.1 \
  --save-steps 1696 \
  --skip-final-alias \
  --eval-steps 1696 \
  --eval-max-batches 50 \
  --logging-steps 10 \
  --dataloader-num-workers 4 \
  --fail-on-truncation \
  --seed 81017 \
  --data-seed 20260901 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
  --counterfactual-grounding-weight 0.0 \
  --periodic-metric-weight 0.1 \
  --periodic-pair-rdf-weight 0.1 \
  --periodic-overlap-weight 0.2 \
  --periodic-coordination-weight 0.05 \
  --periodic-exact-triclinic-pbc \
  --periodic-species-margin-scale 0.55 \
  --periodic-species-margin-floor 0.60 \
  --periodic-species-margin-ceiling 1.40 \
  --periodic-relation-rank 64 \
  "${gate_args[@]}" \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-1696"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" ]]
[[ -f "${CHECKPOINT}/periodic_relation_config.json" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoint_dirs < <(find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -printf '%f\n' | sort)
[[ "${#checkpoint_dirs[@]}" -eq 1 && "${checkpoint_dirs[0]}" == step-1696 ]]

"${PYTHON}" - "${OUTPUT}" "${BASE}" "${METHOD}" <<'PY'
import json, math, pathlib, sys
output=pathlib.Path(sys.argv[1]); base=str(pathlib.Path(sys.argv[2])); method=sys.argv[3]
config=json.load((output/'run_config.json').open())
assert config['checkpoint_path']==base and config['max_train_steps']==1696
assert config['seed']==81017 and config['data_seed']==20260901
assert config['distributed'] is True and config['world_size']==2
assert config['batch_size']==1 and config['grad_accum']==8
assert config['periodic_exact_triclinic_pbc'] is True
assert config['periodic_species_margin_scale']==0.55
assert config['periodic_species_margin_floor']==0.60
assert config['periodic_species_margin_ceiling']==1.40
assert config['periodic_relation_uncertainty_gate'] is (method=='B')
relation=json.load((output/'checkpoints/step-1696/periodic_relation_config.json').open())
assert relation['step0_checked'] is True and relation['step0_max_logit_delta']==0.0
assert relation['config']['uncertainty_gate'] is (method=='B')
events=[json.loads(line) for line in (output/'training_log.jsonl').open() if line.strip()]
train=[row for row in events if row.get('event')=='train']
assert train and int(train[-1]['step'])==1696
assert any(row.get('event')=='eval' and int(row['step'])==1696 for row in events)
for row in train:
    for key in ('loss','task_loss','grad_norm','lr','periodic_relation_grad_norm',
                'periodic_relation_output_grad_norm','periodic_geometry_metric_loss',
                'periodic_geometry_pair_rdf_loss','periodic_geometry_overlap_loss',
                'periodic_geometry_coordination_loss'):
        assert math.isfinite(float(row[key])), (key,row[key])
PY

sha256sum "${CHECKPOINT}/adapter_model.safetensors" \
  "${CHECKPOINT}/periodic_relation_adapter.pt" \
  "${CHECKPOINT}/periodic_relation_config.json" >"${RUN}/step1696_policy.sha256"
printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors \
  ! -name periodic_relation_adapter.pt -print0 | sort -z | xargs -0 sha256sum \
  >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"


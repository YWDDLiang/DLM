#!/usr/bin/env bash
#SBATCH --job-name=btrd-residual
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=06:00:00
#SBATCH --output=btrd-residual-%j.out
#SBATCH --error=btrd-residual-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export deployed code root}"
readonly DATA="${BTRD_DATA_DIR:?export completed BTRD data root}"
readonly DATA_MANIFEST_SHA="${BTRD_DATA_MANIFEST_SHA256:?pin BTRD manifest}"
readonly TEACHER_AUDIT="${BTRD_TEACHER_AUDIT_DIR:?export passed teacher identity audit}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/g2_full_epoch_A_39164/train/checkpoints/step-1696"
readonly TRAINER="${CODE_ROOT}/src/scripts/llada_sft.py"
readonly RUN="${ROOT}/runs/btrd_residual_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly POLICY_SHA=712c941922611ea3640b171319c8ae9a4cbdbf31f83cdba6f61046b9de61bfce
readonly RELATION_SHA=675d68b9e9c0aeb7f3383050b8c85716aa2f7c9303d5a8e1e75b0761db54b9f8

export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error(){
  rc="$1"; line="$2"; command="$3"; set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${DATA}/_SUCCESS" && ! -e "${DATA}/_FAILED" ]]
[[ -f "${TEACHER_AUDIT}/_SUCCESS" && ! -e "${TEACHER_AUDIT}/_FAILED" ]]
"${PYTHON}" - "${TEACHER_AUDIT}/BTRD_TEACHER_ORDER_AUDIT.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report['status']=='pass' and report['ordered_species_mismatches']==0
assert report['proposal_indices_unique_in_range'] is True
assert report['refined_indices_unique_in_range'] is True
PY
[[ "$(sha256sum "${DATA}/manifest.json" | awk '{print $1}')" == "${DATA_MANIFEST_SHA}" ]]
[[ "$(sha256sum "${POLICY}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY_SHA}" ]]
[[ "$(sha256sum "${POLICY}/periodic_relation_adapter.pt" | awk '{print $1}')" == "${RELATION_SHA}" ]]
IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#allocated_gpus[@]}" -eq 2 ]]

code_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD 2>/dev/null || printf unrecorded)"
cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	btrd_residual_training_v1
initialization	G2-PBC-R_step1696
trainable	periodic_relation_adapter_only
base_policy_frozen	true
data	8192_train_only_tau200_plus_anchor
optimizer_steps	512
effective_batch	16
learning_rate	1e-6_cosine_warmup10
loss	CE+metric0.1+pairRDF0.1+overlap0.2+coordination0.05+transport0.25
training_seed	81017
checkpoint_selection	none_step512_only
planner_resampled	false
energy_or_hull_label	false
code_commit	${code_commit}
EOF
sha256sum "${DATA}/manifest.json" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${POLICY}/adapter_model.safetensors" "${POLICY}/periodic_relation_adapter.pt" \
  "${TEACHER_AUDIT}/BTRD_TEACHER_ORDER_AUDIT.json" "${TRAINER}" \
  "${RUN}/scientific_contract.tsv" >"${RUN}/INPUTS.sha256"

torchrun --standalone --master-port 29638 --nproc_per_node=2 "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${POLICY}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 512 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-6 \
  --lr-scheduler cosine \
  --warmup-steps 10 \
  --min-lr-ratio 0.1 \
  --save-steps 512 \
  --skip-final-alias \
  --eval-steps 512 \
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
  --periodic-metric-weight 0.1 \
  --periodic-pair-rdf-weight 0.1 \
  --periodic-overlap-weight 0.2 \
  --periodic-coordination-weight 0.05 \
  --basin-transport-weight 0.25 \
  --periodic-exact-triclinic-pbc \
  --periodic-image-radius 2 \
  --periodic-species-margin-scale 0.55 \
  --periodic-species-margin-floor 0.60 \
  --periodic-species-margin-ceiling 1.40 \
  --periodic-relation-rank 64 \
  --periodic-relation-checkpoint "${POLICY}" \
  --periodic-relation-circular-min-resultant 0.25 \
  --periodic-relation-only \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly CHECKPOINT="${RUN}/train/checkpoints/step-512"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" ]]
[[ -f "${CHECKPOINT}/periodic_relation_config.json" ]]
[[ -f "${RUN}/train/parameter_partition.json" ]]
[[ ! -e "${RUN}/train/final" ]]

"${PYTHON}" - "${RUN}" "${DATA}" "${POLICY}" <<'PY'
import hashlib, json, math, pathlib, sys
run=pathlib.Path(sys.argv[1]); data=pathlib.Path(sys.argv[2]); policy=pathlib.Path(sys.argv[3])
out=run/'train'; checkpoint=out/'checkpoints/step-512'
config=json.load((out/'run_config.json').open())
partition=json.load((out/'parameter_partition.json').open())
events=[json.loads(line) for line in (out/'training_log.jsonl').open() if line.strip()]
train=[row for row in events if row.get('event')=='train']
assert config['periodic_relation_only'] is True
assert config['max_train_steps']==512 and config['basin_transport_weight']==0.25
assert config['checkpoint_path']==str(policy) and config['periodic_relation_checkpoint']==str(policy)
assert partition['trainable_parameters']>0 and partition['frozen_parameters']>partition['trainable_parameters']
assert train and int(train[-1]['step'])==512
assert any(row.get('event')=='eval' and int(row['step'])==512 for row in events)
for row in train:
    for key in ('loss','task_loss','grad_norm','lr','periodic_relation_grad_norm',
                'periodic_relation_output_grad_norm','periodic_geometry_metric_loss',
                'periodic_geometry_pair_rdf_loss','periodic_geometry_overlap_loss',
                'periodic_geometry_coordination_loss','basin_transport_loss'):
        assert math.isfinite(float(row[key])), (key,row.get(key))
checkpoints=sorted(path.name for path in (out/'checkpoints').iterdir() if path.is_dir())
assert checkpoints==['step-512']
manifest=json.load((data/'manifest.json').open())
payload={
    'schema':'btrd_residual_training_final_v1',
    'status':'complete',
    'optimizer_updates':512,
    'trainable':'periodic_relation_adapter_only',
    'train_rows':manifest['train_rows'],
    'effective_tau200_rows':manifest['effective_tau200_rows'],
    'anchor_or_fallback_rows':manifest['anchor_or_fallback_rows'],
    'checkpoint':str(checkpoint),
    'checkpoint_sha256':hashlib.sha256((checkpoint/'periodic_relation_adapter.pt').read_bytes()).hexdigest(),
    'parameter_partition':partition,
}
(run/'BTRD_TRAIN_FINAL.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

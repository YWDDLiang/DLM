#!/usr/bin/env bash
#SBATCH --job-name=canonical-g2-pbc-r
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=06:00:00
#SBATCH --output=canonical-g2-pbc-r-%j.out
#SBATCH --error=canonical-g2-pbc-r-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly DATA="${CANONICAL_SFT_DATA:?}"
readonly BASE="${CANONICAL_DLM_BASE:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${CODE}/src/scripts/llada_sft.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/canonical_g2_pbc_r_${SLURM_JOB_ID}"

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

[[ -f "${DATA}/_SUCCESS" && -f "${DATA}/vocab_tokens.txt" ]]
[[ "$(wc -l < "${DATA}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l < "${DATA}/val.jsonl")" -eq 9047 ]]
[[ -f "${BASE}/adapter_model.safetensors" ]]
[[ ! -e "${BASE}/periodic_relation_adapter.pt" ]]
"${PYTHON}" - "${DATA}/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
assert x["schema"] == "c3fd_native_teacher_sft_canonical_site_v1"
assert x["site_order"] == "plan_expanded_stable_v1"
assert x["train"]["rows"] == 27136 and x["val"]["rows"] == 9047
assert x["train"]["dropped_rows"] == x["val"]["dropped_rows"] == 0
assert x["outcomes_read"] is False
PY
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	canonical_g2_pbc_r_v1
initialization	${BASE}
teacher	original_MP20_canonical_site
training_seed	81017
data_seed	20260901
optimizer_steps	1696
source_epochs	1
effective_batch	16
learning_rate	5e-6_cosine_warmup25_min0.1
mask_policy	dynamic_geometry_only
loss_weights	metric0.1_pairRDF0.1_overlap0.2_coordination0.05
minimum_image	strict_triclinic_125_image
species_margin	clamp_0.55_radius_sum_0.60A_1.40A_normalized
relation_adapter	rank64_two_layer_zero_initialized
uncertainty_gate	false
checkpoint_selection	none_step1696_only
EOF

CUDA_VISIBLE_DEVICES="${allocated[0]},${allocated[1]}" PYTHONPATH="${CODE}/src" \
  torchrun --standalone --master-port 33817 --nproc_per_node=2 "${TRAINER}" \
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
  --periodic-image-radius 2 \
  --periodic-species-margin-scale 0.55 \
  --periodic-species-margin-floor 0.60 \
  --periodic-species-margin-ceiling 1.40 \
  --periodic-relation-rank 64 \
  --periodic-relation-circular-min-resultant 0.25 \
  --periodic-relation-uncertainty-floor 0.25 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-1696"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" ]]
[[ -f "${CHECKPOINT}/periodic_relation_config.json" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoints < <(
  find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'step-*' -printf '%f\n' | sort
)
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-1696 ]]
"${PYTHON}" - "${OUTPUT}" "${BASE}" <<'PY'
import json,math,pathlib,sys
output=pathlib.Path(sys.argv[1]); base=str(pathlib.Path(sys.argv[2]))
config=json.load((output/"run_config.json").open())
assert config["checkpoint_path"] == base
assert config["max_train_steps"] == 1696 and config["world_size"] == 2
assert config["seed"] == 81017 and config["data_seed"] == 20260901
assert config["periodic_exact_triclinic_pbc"] is True
assert config["periodic_image_radius"] == 2
assert config["periodic_species_margin_scale"] == 0.55
assert config["periodic_species_margin_floor"] == 0.60
assert config["periodic_species_margin_ceiling"] == 1.40
assert config["periodic_relation_uncertainty_gate"] is False
relation=json.load((output/"checkpoints/step-1696/periodic_relation_config.json").open())
assert relation["step0_checked"] is True and relation["step0_max_logit_delta"] == 0.0
assert relation["config"]["image_radius"] == 2
assert relation["config"]["uncertainty_gate"] is False
events=[json.loads(line) for line in (output/"training_log.jsonl").open() if line.strip()]
train=[row for row in events if row.get("event") == "train"]
assert train and int(train[-1]["step"]) >= 1690
assert any(row.get("event") == "eval" and int(row["step"]) == 1696 for row in events)
for row in train:
    for key in ("loss","task_loss","grad_norm","lr","periodic_relation_grad_norm",
                "periodic_relation_output_grad_norm","periodic_geometry_metric_loss",
                "periodic_geometry_pair_rdf_loss","periodic_geometry_overlap_loss",
                "periodic_geometry_coordination_loss"):
        assert math.isfinite(float(row[key])), (key,row[key])
PY

printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

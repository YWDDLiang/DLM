#!/usr/bin/env bash
#SBATCH --job-name=compact-v2-g2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=170G
#SBATCH --time=06:00:00
#SBATCH --output=compact-v2-g2-%j.out
#SBATCH --error=compact-v2-g2-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly BASE="${ROOT}/runs/c3fd_native_teacher_sft_38703/train_seed82017/checkpoints/step-3392"
readonly TRAINER="${CODE_ROOT}/src/scripts/llada_sft.py"
readonly OBJECTIVE="${CODE_ROOT}/src/crystal_dlm/periodic_geometry_objective.py"
readonly ADAPTER="${CODE_ROOT}/src/crystal_dlm/periodic_relation_adapter.py"
readonly RUNTIME="${CODE_ROOT}/src/crystal_dlm/periodic_relation_runtime.py"
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

readonly MODEL_TREE_SHA=812a0ef72b2b2290b19b8beac32fd8ba0f4d32f308268f0c7914c4efd8755037
readonly DATA_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly DATA_TRAIN_SHA=f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068
readonly DATA_VAL_SHA=bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1
readonly BASE_ADAPTER_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20
readonly TRAINER_SHA=cc9ba7418293a9b8245d9a06236a9b3f2ed9834a6226a4a8cc1540d2e5c791e9
readonly OBJECTIVE_SHA=b4d1df1b171fba9554355fe2c67384e96fbd6d99b533d39d5bb5a56c698b5c1d
readonly ADAPTER_SHA=a72c47e549b841e501741cd2d8dccafa3b238e89496cb45dabdd0f7b704d9c8d
readonly RUNTIME_SHA=ec6f444456bd40931c2afe956f8b5c92b4c43e61733b34549fd8422a3e3a3bd3

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

actual_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${CODE_COMMIT}" ]] || exit 2
[[ -f "${DATA}/_SUCCESS" && -f "${BASE}/adapter_model.safetensors" ]] || exit 2
for spec in \
  "${DATA_MANIFEST_SHA}:${DATA}/manifest.json" \
  "${DATA_TRAIN_SHA}:${DATA}/train.jsonl" \
  "${DATA_VAL_SHA}:${DATA}/val.jsonl" \
  "${BASE_ADAPTER_SHA}:${BASE}/adapter_model.safetensors" \
  "${TRAINER_SHA}:${TRAINER}" \
  "${OBJECTIVE_SHA}:${OBJECTIVE}" \
  "${ADAPTER_SHA}:${ADAPTER}" \
  "${RUNTIME_SHA}:${RUNTIME}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
actual_model_tree_sha="$(find "${MODEL}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${actual_model_tree_sha}" == "${MODEL_TREE_SHA}" ]] || exit 2
IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
[[ "${#allocated_gpus[@]}" -eq 1 ]] || exit 2

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_v1
initialization	${BASE}
base_adapter_sha256	${BASE_ADAPTER_SHA}
code_commit	${CODE_COMMIT}
training_seed	81017
data_seed	20260901
optimizer_steps	348
checkpoint_selection	none_step348_only
world_size	1
effective_batch	16
learning_rate	5e-6_cosine_warmup25_min0.1
mask_policy	dynamic_geometry_only
G1_loss_weights	metric0.1_pairRDF0.1_overlap0.2_coordination0.05
relation_adapter	rank64_two_layer_species_pair_RBF_zero_Wout
relation_forward	acyclic_q0_soft_geometry_residual_q1
decoder_change	false
development_full_endpoint_required	true
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${DATA}/manifest.json" \
  "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${BASE}/adapter_model.safetensors" "${TRAINER}" "${OBJECTIVE}" \
  "${ADAPTER}" "${RUNTIME}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 348 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 25 \
  --min-lr-ratio 0.1 \
  --save-steps 348 \
  --skip-final-alias \
  --eval-steps 348 \
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
  --periodic-relation-rank 64 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-348"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]] || exit 3
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" ]] || exit 3
[[ -f "${CHECKPOINT}/periodic_relation_config.json" ]] || exit 3
[[ ! -e "${OUTPUT}/final" ]] || exit 3
mapfile -t checkpoint_dirs < <(find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -printf '%f\n' | sort)
[[ "${#checkpoint_dirs[@]}" -eq 1 && "${checkpoint_dirs[0]}" == step-348 ]] || exit 3

"${PYTHON}" - "${OUTPUT}" "${BASE}" <<'PY'
import json, math, pathlib, sys, torch
output = pathlib.Path(sys.argv[1]); base = str(pathlib.Path(sys.argv[2]))
config = json.load((output / "run_config.json").open())
assert config["checkpoint_path"] == base
assert config["periodic_relation_rank"] == 64
assert config["periodic_relation_checkpoint"] is None
assert config["dynamic_geometry_only"] is True
assert config["max_train_steps"] == 348 and config["seed"] == 81017
relation = json.load((output / "checkpoints/step-348/periodic_relation_config.json").open())
assert relation["step0_checked"] is True and relation["step0_max_logit_delta"] == 0.0
assert relation["config"]["rank"] == 64 and relation["config"]["max_sites"] == 20
state = torch.load(output / "checkpoints/step-348/periodic_relation_adapter.pt", map_location="cpu", weights_only=True)
assert float(state["output_projection.weight"].abs().max()) > 0.0
events = [json.loads(line) for line in (output / "training_log.jsonl").open() if line.strip()]
train = [row for row in events if row.get("event") == "train"]
assert train and any(row.get("event") == "eval" and int(row["step"]) == 348 for row in events)
early = [row for row in train if int(row["step"]) <= 10]
assert {int(row["step"]) for row in early} == set(range(1, 11))
assert float(early[0]["periodic_relation_output_grad_norm"]) > 0.0
for row in train:
    for key in (
        "loss", "task_loss", "grad_norm", "lr", "periodic_relation_grad_norm",
        "periodic_relation_output_grad_norm", "periodic_geometry_metric_loss",
        "periodic_geometry_pair_rdf_loss", "periodic_geometry_overlap_loss",
        "periodic_geometry_coordination_loss",
    ):
        assert math.isfinite(float(row[key])), (key, row[key])
PY

sha256sum "${CHECKPOINT}/adapter_model.safetensors" \
  "${CHECKPOINT}/periodic_relation_adapter.pt" \
  "${CHECKPOINT}/periodic_relation_config.json" >"${RUN}/step348_policy.sha256"
printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors \
  ! -name periodic_relation_adapter.pt -print0 | sort -z | xargs -0 sha256sum \
  >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"

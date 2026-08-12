#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-b0d1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/h1a2-b0d1-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5/logs/h1a2-b0d1-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
SOURCE="$RUN_ROOT/source"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUNTIME="$R03E/runtime"
BODY_SCRIPT="$RUNTIME/scripts/sample_llada_h1a2_factorial_body.py"
REFINE_SCRIPT="$RUNTIME/scripts/refine_h1a2_factorial_with_crysllmgen.py"
LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
COHORT_ROOT="$RUN_ROOT/planner/retrained_seed17_world2_b4_topology_match"
COHORT="$COHORT_ROOT/frozen/cohort256.jsonl"
RAW="$COHORT_ROOT/raw/raw_generations.jsonl"
ROOT="$RUN_ROOT/historical_h1a2_dlm_control"
BASE_MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
BODY_CHECKPOINT="$PROJECT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
PLANNER_MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

EXIT_CODE="$RUN_ROOT/status/h1a2_dlm_control_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/h1a2_dlm_control_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 16
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$LEDGER" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f
test "$(sha256sum "$BODY_SCRIPT" | cut -d' ' -f1)" = 3e1fdca67d070a72fa2ae031d8e65a3382960e76f0c3a3d666e1d30c12889892
test "$(sha256sum "$REFINE_SCRIPT" | cut -d' ' -f1)" = 7b4a98224f84df38f5b9a7ae881430f3e732a03f292cbe3b788fc9924e3c70c1
test "$(sha256sum "$RUNTIME/crystal_dlm/h1a2_factorial_contract.py" | cut -d' ' -f1)" = 2717c2ab384eec91b7730da3c59dceabaa36ac8be0c24872d5150b7d88b9474b
test "$(sha256sum "$RUNTIME/crystal_dlm/h1a2_factorial_runtime.py" | cut -d' ' -f1)" = e3b371227896a013b6aece71babaa3257f38fde376f3bf878de03f6de7b268e4
test "$(sha256sum "$RUNTIME/scripts/refine_dlm_with_crysllmgen.py" | cut -d' ' -f1)" = a0cb2c54c149aee0a4f36c147d93eca6799e3b17879f76c108721c934f087b2a
test "$(sha256sum "$RUNTIME/reference/crysllmgen/models_ddpm/diffusion.py" | cut -d' ' -f1)" = d27eff3839e1a0da0098a287225e3c4f2239188cbffed3387d37dc8b95bdb6ca
test "$(sha256sum "$BODY_CHECKPOINT/adapter_model.safetensors" | cut -d' ' -f1)" = 5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
test -f "$RUN_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/status/refiner_contract_SUCCESS"
test -f "$COHORT" -a -f "$RAW"
test ! -e "$ROOT"
mkdir -p "$ROOT"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export H1_REFERENCE_ATTEMPT_LEDGER="$LEDGER"
export PYTHONPATH="$SOURCE:$RUNTIME:$R03E"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/h1a2_dlm_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CRYSLLMGEN_METRICS_NUM_CPUS=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 2 or any("A800" not in name for name in names):
    raise SystemExit(f"expected two A800s, observed {names}")
if not metadata.version("smact").startswith("3.1"):
    raise SystemExit("SMACT 3.1 required")
print({"gpu": names, "torch": torch.__version__, "smact": metadata.version("smact")})
PY

echo "STAGE historical_h1a2_dlm adapt_seed17_world2_plans"
PLANNER_ADAPTER_SHA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_upstream"]["adapter_sha256"])' "$SOURCE/CONFIG.json")"
"$PYTHON" "$SOURCE/prepare_h1a2_dlm_inputs.py" \
  --cohort "$COHORT" \
  --raw-generations "$RAW" \
  --base-model "$PLANNER_MODEL" \
  --planner-checkpoint-sha256 "$PLANNER_ADAPTER_SHA" \
  --output-dir "$ROOT/inputs"

echo "STAGE historical_h1a2_dlm B0_d1_body_world2"
torchrun --nproc_per_node=2 --master_port="$((31000 + SLURM_JOB_ID % 9000))" \
  "$SOURCE/run_h1a2_dlm_body.py" \
  --base-script "$BODY_SCRIPT" \
  --base-script-sha256 3e1fdca67d070a72fa2ae031d8e65a3382960e76f0c3a3d666e1d30c12889892 \
  --planner-attempts "$ROOT/inputs/planner_attempts.jsonl" \
  -- \
  --model-path "$BASE_MODEL" \
  --checkpoint-path "$BODY_CHECKPOINT" \
  --checkpoint-sha256 5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d \
  --planner-attempts "$ROOT/inputs/planner_attempts.jsonl" \
  --factorial-arm M00 \
  --output-dir "$ROOT/body" \
  --crysllmgen-dir "$RUNTIME/reference/crysllmgen" \
  --num-samples 256 \
  --base-seed 17 \
  --expected-tokenizer-vocab-sha256 3acc073da85047265769f2dccd93543fa9d7cbfa95021aef54ef282b13ce2f37 \
  --temperature 0.7 \
  --cfg-scale 0.0 \
  --remasking low_confidence \
  --min-lattice-rad 0.0001

echo "STAGE historical_h1a2_dlm model494_refine800_world2"
torchrun --nproc_per_node=2 --master_port="$((40000 + SLURM_JOB_ID % 9000))" \
  "$REFINE_SCRIPT" \
  --body-attempts "$ROOT/body/body_attempts.jsonl" \
  --proposal-graphs "$ROOT/body/proposal_graphs.pt" \
  --factorial-arm M00 \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$ROOT/refinement" \
  --crysllmgen-dir "$RUNTIME/reference/crysllmgen" \
  --num-samples 256

echo "STAGE historical_h1a2_dlm finalize_post_model494"
"$PYTHON" "$SOURCE/finalize_h1a2_dlm_post.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
  --refinement-dir "$ROOT/refinement" \
  --output-dir "$ROOT/post_model494/generation"

echo "STAGE historical_h1a2_dlm post_model494_direct_crysllmgen"
"$PYTHON" "$RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
  --generation-jsonl "$ROOT/post_model494/generation/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root "$RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "$ROOT/post_model494/direct" \
  --eval-model-name mp20

touch "$ROOT/generation_SUCCESS"
touch "$RUN_ROOT/status/h1a2_dlm_control_SUCCESS"
echo "STAGE historical_h1a2_dlm generation_complete"

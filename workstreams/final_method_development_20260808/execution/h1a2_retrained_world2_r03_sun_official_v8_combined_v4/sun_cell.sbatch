#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-sun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%9
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
SOURCE="$RUN_ROOT/source"
UPSTREAM="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
RUNTIME="$R03E/runtime"
LEGACY_CACHE="$PROJECT/runs/20260810_h1_r03_raw_plan_b3_safeaxis_direct_sun_r03g_schema_repair_v5/mp_cache/r03f_common_mp_thermo_snapshot.jsonl"
CHGNET_CACHE="$PROJECT/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
CHGNET_MODEL=/public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar
CHGNET_RUNTIME=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"

INDEX="${SLURM_ARRAY_TASK_ID:?array task required}"
EXIT_CODE="$RUN_ROOT/status/preliminary_cell_${INDEX}_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/preliminary_cell_${INDEX}_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test -f "$RUN_ROOT/status/preparation_SUCCESS"
test -f "$UPSTREAM/status/generation_assembly_SUCCESS"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$R03E/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4
test "$(sha256sum "$RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" | cut -d' ' -f1)" = e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de
test "$(sha256sum "$RUNTIME/crystal_dlm/wqcodiff/crysllmgen/a100_sun.py" | cut -d' ' -f1)" = 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968
test "$(sha256sum "$LEGACY_CACHE" | cut -d' ' -f1)" = 56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917
test "$(sha256sum "$CHGNET_CACHE" | cut -d' ' -f1)" = 9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71
test "$(sha256sum "$CHGNET_MODEL" | cut -d' ' -f1)" = d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1
test "$(sha256sum "$CHGNET_RUNTIME" | cut -d' ' -f1)" = d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1
mapfile -t CELL_VALUES < <("$PYTHON" "$SOURCE/cell_descriptor.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --cell-index "$INDEX")
test "${#CELL_VALUES[@]}" -eq 4
CELL_ID="${CELL_VALUES[0]}"
PANEL="${CELL_VALUES[1]}"
COHORT="${CELL_VALUES[2]}"
GENERATION="${CELL_VALUES[3]}"
STAGE=post_model494
OUTPUT="$RUN_ROOT/preliminary/$(printf '%03d' "$INDEX")_${CELL_ID}"
test ! -e "$OUTPUT"
test -f "$GENERATION"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/preliminary_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected one A800, observed {names}")
if metadata.version("torch-scatter") != "2.1.2+pt24cu121":
    raise SystemExit("registered torch-scatter build changed")
print({"gpu": names, "torch": torch.__version__, "torch_scatter": metadata.version("torch-scatter")})
PY

echo "STAGE cell=$INDEX cell_id=$CELL_ID panel=$PANEL cohort=$COHORT stage=$STAGE freeze_un_chgnet"
export PYTHONPATH="$SOURCE:$RUNTIME:$R03E"
"$PYTHON" "$SOURCE/run_frozen_official.py" \
  --script run_completed_cache_sun.py \
  --script-sha256 585496be8fde0a50bc09f178ba456c50cfed4e2e612ef7678a3be09106936937 \
  -- \
  --runner "$RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
  --runner-sha256 e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de \
  --a100-sun-module-sha256 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968 \
  --expected-mp-hull-cache-sha256 56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917 \
  -- \
  --generation-jsonl "$GENERATION" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache "$LEGACY_CACHE" \
  --chgnet-relax-cache "$CHGNET_CACHE" \
  --chgnet-model-asset "$CHGNET_MODEL" \
  --chgnet-runtime-checkpoint "$CHGNET_RUNTIME" \
  --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
  --execution-patch-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --output-dir "$OUTPUT"

echo "STAGE cell=$INDEX cell_id=$CELL_ID panel=$PANEL stage=$STAGE validate_preliminary"
"$PYTHON" "$SOURCE/validate_preliminary.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN_ROOT" \
  --cell-index "$INDEX"
touch "$RUN_ROOT/status/preliminary_cell_${INDEX}_SUCCESS"
echo "STAGE cell=$INDEX cell_id=$CELL_ID panel=$PANEL stage=$STAGE preliminary_complete"

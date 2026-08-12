#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-tref
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7/logs/topology-refine-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7/logs/topology-refine-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7"
SOURCE="$RUN_ROOT/source"
V4="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4/body_source"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
R03E_RUNTIME="$R03E/runtime"
LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

REPEAT="${SLURM_ARRAY_TASK_ID:?array task required}"
case "$REPEAT" in 0|1|2|3) ;; *) echo "invalid topology refine task" >&2; exit 2 ;; esac
ROOT="$RUN_ROOT/topology_match"
INPUT="$ROOT/process_inputs/repeat_$REPEAT"
CELL="$ROOT/processes/repeat_$REPEAT"
EXIT_CODE="$RUN_ROOT/status/topology_refine_${REPEAT}_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/topology_refine_${REPEAT}_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$LEDGER" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f
test "$(sha256sum "$V4/refine1000.py" | cut -d' ' -f1)" = 12c5db21a15b7245d00b4388e2f5204a7dd91b3e10af9831e2a18b347f6cf3a6
test "$(sha256sum "$V4/finalize_post1000.py" | cut -d' ' -f1)" = 95d664e206166a332824e0b1639365ebde48f00e26a60ed7724b37878f29a6c8
test "$(sha256sum "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" | cut -d' ' -f1)" = 556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
test "$(sha256sum "$R03E/refine_safeaxis_arm.py" | cut -d' ' -f1)" = 71ab6a55a793a991c8a677c285f9cad1a79bf4c1667ccedf4c5c5199b4092f54
test "$(sha256sum "$R03E_RUNTIME/scripts/refine_dlm_with_crysllmgen.py" | cut -d' ' -f1)" = a0cb2c54c149aee0a4f36c147d93eca6799e3b17879f76c108721c934f087b2a
test "$(sha256sum "$R03E_RUNTIME/reference/crysllmgen/models_ddpm/diffusion.py" | cut -d' ' -f1)" = d27eff3839e1a0da0098a287225e3c4f2239188cbffed3387d37dc8b95bdb6ca
test -f "$REFINER"
test "$(stat -c %s "$REFINER")" = 147645242
test -f "$RUN_ROOT/status/topology_body_SUCCESS"
test -f "$RUN_ROOT/status/refiner_contract_SUCCESS"
test -f "$ROOT/process_inputs/projection_SUCCESS"
test -f "$ROOT/process_inputs/projection_report.json"
test -f "$INPUT/body_attempts.jsonl"
test -f "$INPUT/proposal_graphs.pt"
test ! -e "$CELL"
mkdir -p "$CELL"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export H1_REFERENCE_ATTEMPT_LEDGER="$LEDGER"
export H1_RECOVERY_PANEL=topology_match
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/topology_refine_${SLURM_ARRAY_JOB_ID}_${REPEAT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CRYSLLMGEN_METRICS_NUM_CPUS=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise SystemExit(f"expected one A800, observed {names}")
if not metadata.version("smact").startswith("3.1"):
    raise SystemExit("SMACT 3.1 required")
print({"gpu": names, "torch": torch.__version__, "smact": metadata.version("smact")})
PY

echo "STAGE topology_match repeat=$REPEAT model494_refine800"
export PYTHONPATH="$SOURCE:$R03E_RUNTIME:$R03E"
"$PYTHON" "$SOURCE/run_frozen_v4.py" \
  --script "$V4/refine1000.py" \
  --script-sha256 12c5db21a15b7245d00b4388e2f5204a7dd91b3e10af9831e2a18b347f6cf3a6 \
  -- \
  --arm R03 \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --body-attempts "$INPUT/body_attempts.jsonl" \
  --proposal-graphs "$INPUT/proposal_graphs.pt" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$CELL/refinement" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen"

echo "STAGE topology_match repeat=$REPEAT finalize_post_model494"
"$PYTHON" "$SOURCE/run_frozen_v4.py" \
  --script "$V4/finalize_post1000.py" \
  --script-sha256 95d664e206166a332824e0b1639365ebde48f00e26a60ed7724b37878f29a6c8 \
  -- \
  --arm R03 \
  --repeat "$REPEAT" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
  --refinement-dir "$CELL/refinement" \
  --output-dir "$CELL/post_model494/generation"

echo "STAGE topology_match repeat=$REPEAT post_model494_direct_crysllmgen"
"$PYTHON" "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
  --generation-jsonl "$CELL/post_model494/generation/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root "$R03E_RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "$CELL/post_model494/direct" \
  --eval-model-name mp20

touch "$CELL/generation_SUCCESS"
touch "$RUN_ROOT/status/topology_refine_${REPEAT}_SUCCESS"
echo "STAGE topology_match repeat=$REPEAT generation_complete"

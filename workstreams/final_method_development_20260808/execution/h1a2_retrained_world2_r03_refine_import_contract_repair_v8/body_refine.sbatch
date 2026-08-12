#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-r03
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8/logs/body-refine-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8/logs/body-refine-%A_%a.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
UPSTREAM_PLANNER_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
UPSTREAM_BODY_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7"
SOURCE="$RUN_ROOT/source"
V4="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4/body_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
R03E_RUNTIME="$R03E/runtime"
LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

case "${SLURM_ARRAY_TASK_ID:?array task required}" in
  0) INDEX=0; COHORT=retrained_seed52021_world2_b4; COHORT_SHA=b5a897c029947e1cf88d3abbd2b48efed4cfc1694e4dc73ab9a447ce9f3d178c ;;
  1) INDEX=1; COHORT=retrained_seed62023_world2_b4; COHORT_SHA=3913bae3e07d6a06902dc15e8ff8f484cdf32754b204d5fdaeb86b176c663dd9 ;;
  2) INDEX=2; COHORT=retrained_seed72031_world2_b4; COHORT_SHA=daf5b2fa9ec0e00929a91521fac5978e0d8d55008fcf6a2da93ca01224d44b5b ;;
  3) INDEX=3; COHORT=retrained_seed82037_world2_b4; COHORT_SHA=63d764ecb63dd522ffd0fe1844298461fd9525e2169d38c57f2a79e8e8a9c93f ;;
  *) echo "invalid body/refine task" >&2; exit 2 ;;
esac

COHORT_FILE="$UPSTREAM_PLANNER_ROOT/planner/$COHORT/frozen/cohort256.jsonl"
UPSTREAM_CELL="$UPSTREAM_BODY_ROOT/cells/$COHORT"
CELL="$RUN_ROOT/cells/$COHORT"
EXIT_CODE="$RUN_ROOT/status/cell_${INDEX}_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/cell_${INDEX}_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$LEDGER" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f
test "$(sha256sum "$V4/run_body_safeaxis1000.py" | cut -d' ' -f1)" = d569d56bc58c8df13a90c7f41847c991b6c6dd671bf1cd6f99fbbe822e50892b
test "$(sha256sum "$V4/refine1000.py" | cut -d' ' -f1)" = 12c5db21a15b7245d00b4388e2f5204a7dd91b3e10af9831e2a18b347f6cf3a6
test "$(sha256sum "$V4/finalize_post1000.py" | cut -d' ' -f1)" = 95d664e206166a332824e0b1639365ebde48f00e26a60ed7724b37878f29a6c8
test "$(sha256sum "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" | cut -d' ' -f1)" = 556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
test "$(sha256sum "$R03E/refine_safeaxis_arm.py" | cut -d' ' -f1)" = 71ab6a55a793a991c8a677c285f9cad1a79bf4c1667ccedf4c5c5199b4092f54
test "$(sha256sum "$R03E_RUNTIME/scripts/refine_dlm_with_crysllmgen.py" | cut -d' ' -f1)" = a0cb2c54c149aee0a4f36c147d93eca6799e3b17879f76c108721c934f087b2a
test "$(sha256sum "$R03E_RUNTIME/reference/crysllmgen/models_ddpm/diffusion.py" | cut -d' ' -f1)" = d27eff3839e1a0da0098a287225e3c4f2239188cbffed3387d37dc8b95bdb6ca
test "$(sha256sum "$R03D/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f
test "$(sha256sum "$R03E/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4
test -f "$REFINER"
test "$(stat -c %s "$REFINER")" = 147645242
test -f "$UPSTREAM_PLANNER_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/status/body_import_isolation_SELF_TEST_SUCCESS"
test -f "$RUN_ROOT/status/body_runtime_contract_SUCCESS"
test -f "$RUN_ROOT/status/body_runtime_contract_report.json"
test -f "$RUN_ROOT/status/refiner_contract_SUCCESS"
test -f "$RUN_ROOT/status/refiner_contract_report.json"
test -f "$RUN_ROOT/status/upstream_body_evidence_SUCCESS"
test "$(sha256sum "$UPSTREAM_PLANNER_ROOT/planner_terminal_report.json" | cut -d' ' -f1)" = 5a9676dcc10a4c4938d29aa39bbe509a4635f08f2215d49f756ea165729ca966
test -f "$COHORT_FILE"
test "$(sha256sum "$COHORT_FILE" | cut -d' ' -f1)" = "$COHORT_SHA"
test ! -e "$CELL"
test -d "$UPSTREAM_CELL/body"
test -f "$UPSTREAM_CELL/body/body_attempts.jsonl"
test -f "$UPSTREAM_CELL/body/proposal_graphs.pt"
test -f "$UPSTREAM_CELL/body/generation_report.json"
test ! -e "$UPSTREAM_CELL/refinement"
test ! -e "$UPSTREAM_CELL/post_model494"
mkdir -p "$CELL"
cp -a "$UPSTREAM_CELL/body" "$CELL/body"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export H1_REFERENCE_ATTEMPT_LEDGER="$LEDGER"
export H1_RECOVERY_PANEL=fresh_cohort
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/cell_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
version = metadata.version("smact")
if not version.startswith("3.1"):
    raise SystemExit(f"SMACT 3.1 required, observed {version}")
print({"gpu": names, "torch": torch.__version__, "smact": version})
PY

echo "STAGE cohort=$COHORT reuse_v7_body256 model494_refine800"
export PYTHONPATH="$SOURCE:$R03E_RUNTIME:$R03E:$R03D:$PROJECT"
"$PYTHON" "$SOURCE/run_frozen_v4.py" \
  --script "$V4/refine1000.py" \
  --script-sha256 12c5db21a15b7245d00b4388e2f5204a7dd91b3e10af9831e2a18b347f6cf3a6 \
  -- \
  --arm R03 \
  --repeat "$INDEX" \
  --config "$SOURCE/CONFIG.json" \
  --body-attempts "$CELL/body/body_attempts.jsonl" \
  --proposal-graphs "$CELL/body/proposal_graphs.pt" \
  --checkpoint "$REFINER" \
  --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
  --output-dir "$CELL/refinement" \
  --crysllmgen-dir "$R03E_RUNTIME/reference/crysllmgen"

echo "STAGE cohort=$COHORT finalize_post_model494"
"$PYTHON" "$SOURCE/run_frozen_v4.py" \
  --script "$V4/finalize_post1000.py" \
  --script-sha256 95d664e206166a332824e0b1639365ebde48f00e26a60ed7724b37878f29a6c8 \
  -- \
  --arm R03 \
  --repeat "$INDEX" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
  --refinement-dir "$CELL/refinement" \
  --output-dir "$CELL/post_model494/generation"

echo "STAGE cohort=$COHORT post_model494_direct_crysllmgen"
export PYTHONPATH="$SOURCE:$R03E_RUNTIME:$R03E"
"$PYTHON" "$R03E_RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
  --generation-jsonl "$CELL/post_model494/generation/generation.jsonl" \
  --gt-csv reference/crysllmgen/data/mp_20/test.csv \
  --snapshot-root "$R03E_RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "$CELL/post_model494/direct" \
  --eval-model-name mp20

touch "$CELL/generation_SUCCESS"
touch "$RUN_ROOT/status/cell_${INDEX}_generation_SUCCESS"
echo "STAGE cohort=$COHORT generation_complete"

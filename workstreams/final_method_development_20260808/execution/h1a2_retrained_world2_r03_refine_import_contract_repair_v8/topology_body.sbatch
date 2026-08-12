#!/bin/bash
#SBATCH --job-name=h1a2-rtrn-tbody
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8/logs/topology-body-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8/logs/topology-body-%j.err

set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
UPSTREAM_PLANNER_ROOT="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
SOURCE="$RUN_ROOT/source"
V4="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4/body_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
R03E="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
LEDGER="$RUN_ROOT/inputs/reference_attempt_ledger.jsonl"
COHORT="$UPSTREAM_PLANNER_ROOT/planner/retrained_seed17_world2_b4_topology_match/frozen/cohort256.jsonl"
ROOT="$RUN_ROOT/topology_match"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

EXIT_CODE="$RUN_ROOT/status/topology_body_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/topology_body_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 8
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
test "$(sha256sum "$LEDGER" | cut -d' ' -f1)" = 24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f
test "$(sha256sum "$V4/run_body_safeaxis1000.py" | cut -d' ' -f1)" = d569d56bc58c8df13a90c7f41847c991b6c6dd671bf1cd6f99fbbe822e50892b
test "$(sha256sum "$R03D/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f
test "$(sha256sum "$R03E/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4
test -f "$UPSTREAM_PLANNER_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/status/body_import_isolation_SELF_TEST_SUCCESS"
test -f "$RUN_ROOT/status/body_runtime_contract_SUCCESS"
test -f "$RUN_ROOT/status/body_runtime_contract_report.json"
test -f "$RUN_ROOT/status/refiner_contract_SUCCESS"
test "$(sha256sum "$UPSTREAM_PLANNER_ROOT/planner_topology_match_audit.json" | cut -d' ' -f1)" = dce0b7cdcb2ad24b3aa7f418e9ede3e3a32689c61e1a7b3cbe34470ac4120b3a
test -f "$COHORT"
test "$(sha256sum "$COHORT" | cut -d' ' -f1)" = 3a3107489866b551069870cac12a40f02b7c6a8f313845c6df91bc450e529f39
test ! -e "$ROOT"
mkdir -p "$ROOT"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export H1_REFERENCE_ATTEMPT_LEDGER="$LEDGER"
export H1_RECOVERY_PANEL=topology_match
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/topology_body_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
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

echo "STAGE topology_match r03_safe_axis_body256_once"
export PYTHONPATH="$SOURCE:$R03D/runtime:$R03D"
"$PYTHON" "$SOURCE/run_body_replay.py" \
  --base-script "$V4/run_body_safeaxis1000.py" \
  --base-script-sha256 d569d56bc58c8df13a90c7f41847c991b6c6dd671bf1cd6f99fbbe822e50892b \
  -- \
  --arm R03 \
  --repeat 0 \
  --cohort "$COHORT" \
  --crysllmgen-dir "$R03E/runtime/reference/crysllmgen" \
  --config "$SOURCE/CONFIG.json" \
  --output-dir "$ROOT/body_realization"

echo "STAGE topology_match project_identical_refiner_process_inputs"
export PYTHONPATH="$SOURCE"
"$PYTHON" "$SOURCE/project_topology_inputs.py" \
  --body-attempts "$ROOT/body_realization/body_attempts.jsonl" \
  --proposal-graphs "$ROOT/body_realization/proposal_graphs.pt" \
  --output-dir "$ROOT/process_inputs"

touch "$RUN_ROOT/status/topology_body_SUCCESS"
echo "STAGE topology_match body_projection_complete"

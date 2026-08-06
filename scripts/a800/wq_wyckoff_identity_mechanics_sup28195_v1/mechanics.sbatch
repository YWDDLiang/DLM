#!/bin/bash
#SBATCH --job-name=wq-wtb-idv2-32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
IDENTITY=wq_wyckoff_identity_mechanics_sup28195_v1
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_identity_mechanics_sup28195_v1.json
CONTRACT_SHA256=6ca0d3f292aff8fcaedd97566fc5b2367bc17c24f6a075b5a47490613cb5663d
AUTHORIZATION=diagnostics/authorization_records/wq_wyckoff_identity_mechanics_sup28195_v1.json
AUTHORIZATION_SHA256=d97e82caa5421a860a1943b50b17effabfd204980e994696041e692467cce417
RUNNER=scripts/a800/run_wq_wyckoff_identity_mechanics_sup28195_v1.py
RUNNER_SHA256=3f5adf540b003d1e70863e8c5610b536c73b3dd3e1724b1f9386b64836d5c467
IDENTITY_MODULE=crystal_dlm/wqcodiff/crysllmgen/wtb_identity_v2.py
IDENTITY_MODULE_SHA256=af98a0bdb0c919748afbbef267139e5743fda78b41fabc086e06e966e7222435
LEGACY_ARMS=scripts/a800/run_wq_wyckoff_chart_retraction_arms256_v1.py
LEGACY_ARMS_SHA256=056ea0406154090243c13fd3010cfed07a205f0aa0ec2985665fa2c12dfd7121
GLOBAL_RETRACTION=scripts/a800/run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py
GLOBAL_RETRACTION_SHA256=63925aa0b877914b35240e55459026cf10e660665a14304a427c20253ef57a35
TANGENT_ENGINE=scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py
TANGENT_ENGINE_SHA256=edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3
SCHEDULE_BRIDGE=scripts/a800/run_wq_schedule_correct_bridge_parity_v1.py
SCHEDULE_BRIDGE_SHA256=22dd1ad3884fd6bf7fc8832e12372cecba53c35e89a2eca977fef121177f310b
RUNTIME=crystal_dlm/wqcodiff/runtime.py
RUNTIME_SHA256=8b5ba104ee1be25ff7f8a14b703193b33920bfd71abd52b1ba1e0d082e909ea4
STATE=crystal_dlm/wqcodiff/state.py
STATE_SHA256=75aaa737bc8ac992318b4b1a98a5300608a054cc7ef629899fd9e27c55b47d0f
SOURCE_ROOT="$RUN/outputs/wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/sources"
SOURCE="$SOURCE_ROOT/source_attempts.jsonl"
SOURCE_SHA256=3246a24d2595ae760e15f402222d6730a2a0fdbc404636254a7fa995559d56f2
SOURCE_REPORT="$SOURCE_ROOT/source_report.json"
SOURCE_REPORT_SHA256=9c6d5a9f2570f73b87ffa4c2bac898499588ea803cdf5803103408ce22323be9
JOB28195_AUDIT=runs/remote_audit/20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/terminal_failure_audit_job28195.json
JOB28195_AUDIT_SHA256=124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9
PARENT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
PARENT_SHA256=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
OUTPUT="$RUN/outputs/$IDENTITY"

: "${WTB_IDV2_PATCH_SHA256:?submission must export WTB_IDV2_PATCH_SHA256}"
: "${WTB_IDV2_AUTHORIZATION_SHA256:?submission must export WTB_IDV2_AUTHORIZATION_SHA256}"
if [[ ! "$WTB_IDV2_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB_IDV2_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$WTB_IDV2_AUTHORIZATION_SHA256" != "$AUTHORIZATION_SHA256" ] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "WTB identity-v2 execution/resource identity changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB_IDV2_PATCH_SHA256}.json"
while read -r path expected; do
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$CONTRACT $CONTRACT_SHA256
$AUTHORIZATION $AUTHORIZATION_SHA256
$RUNNER $RUNNER_SHA256
$IDENTITY_MODULE $IDENTITY_MODULE_SHA256
$LEGACY_ARMS $LEGACY_ARMS_SHA256
$GLOBAL_RETRACTION $GLOBAL_RETRACTION_SHA256
$TANGENT_ENGINE $TANGENT_ENGINE_SHA256
$SCHEDULE_BRIDGE $SCHEDULE_BRIDGE_SHA256
$RUNTIME $RUNTIME_SHA256
$STATE $STATE_SHA256
$SOURCE $SOURCE_SHA256
$SOURCE_REPORT $SOURCE_REPORT_SHA256
$JOB28195_AUDIT $JOB28195_AUDIT_SHA256
$PARENT $PARENT_SHA256
EOF
test ! -e "$OUTPUT"

python3 - "$CONTRACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    contract.get("schema") != "wq_wyckoff_identity_mechanics_contract_v1"
    or contract.get("identity") != "wq_wyckoff_identity_mechanics_sup28195_v1"
    or contract.get("status") != "authorized_development_execution"
    or contract["identity_contract"]["composition_identity"]
    != "element_count_multiset"
    or contract["identity_contract"]["legacy_ordered_atom_identity_role"]
    != "diagnostic_only"
    or contract["identity_contract"]["legacy_ordered_atom_mismatch_blocking"]
    is not False
    or contract["lineage"]["job28195_reinterpreted"] is not False
    or contract["lineage"]["confirmatory_evidence"] is not False
):
    raise SystemExit("WTB identity-v2 contract boundary changed")
for entry in contract["implementation"].values():
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"WTB identity-v2 installed source changed: {path}")
print("wtb_identity_v2_installed_source_hash_gate=PASS")
PY

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

python - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit("WTB identity-v2 requires exactly one visible CUDA device")
name = torch.cuda.get_device_name(0)
if "A800" not in name:
    raise SystemExit(f"WTB identity-v2 requires A800; observed {name}")
print(f"wtb_identity_v2_device={name}")
print(f"wtb_identity_v2_torch={torch.__version__}")
print(f"wtb_identity_v2_cuda={torch.version.cuda}")
print("wtb_identity_v2_environment_smoke=PASS")
PY

usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "pipeline_stage=permutation_safe_source_identity_256"
echo "pipeline_stage=development_R_U_T_mechanics_32"
python "$RUNNER" \
  --contract "$CONTRACT" \
  --source-jsonl "$SOURCE" \
  --source-report "$SOURCE_REPORT" \
  --checkpoint "$PARENT" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$WTB_IDV2_PATCH_SHA256" \
  --device cuda

python - "$OUTPUT/terminal_report.json" "$OUTPUT/source_identity_audit.json" \
  "$CONTRACT_SHA256" "$WTB_IDV2_PATCH_SHA256" <<'PY'
import json
import sys
from pathlib import Path

terminal = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
checks = terminal.get("mechanics_gate_checks", {})
if (
    terminal.get("schema") != "wq_wyckoff_identity_mechanics_terminal_v1"
    or terminal.get("identity") != "wq_wyckoff_identity_mechanics_sup28195_v1"
    or terminal.get("ok") is not True
    or terminal.get("acceptance") != "PASS"
    or terminal.get("confirmatory_evidence") is not False
    or terminal.get("job28195_reinterpreted") is not False
    or terminal.get("automatic_confirmatory_authorized") is not False
    or terminal.get("legacy_ordered_atom_identity_role") != "diagnostic_only"
    or terminal.get("retry_or_replacement_used") is not False
    or source.get("permutation_safe_identity_passed") != 256
    or source.get("permutation_safe_identity_failed") != 0
    or source.get("legacy_order_mismatch_is_blocking") is not False
    or not checks
    or not all(value is True for value in checks.values())
):
    raise SystemExit("WTB identity-v2 terminal acceptance failed")
print("wtb_identity_v2_terminal_acceptance=PASS")
print("wtb_identity_v2_evidence=development_only")
print(
    "wtb_identity_v2_legacy_order_mismatches="
    f"{source['legacy_order_identity_mismatched']}"
)
PY

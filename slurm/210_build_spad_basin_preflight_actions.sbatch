#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-actions
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=spad-basin-actions-%j.out
#SBATCH --error=spad-basin-actions-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly STATES="${ROOT}/data/spad_basin_preflight_states_v1_20260904/states.jsonl"
readonly TRAIN_RUN="${ROOT}/runs/spad_basin_closure_ce_39700"
readonly POLICY="$(<"${TRAIN_RUN}/POLICY_PATH")"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly RUN="${ROOT}/runs/spad_basin_preflight_actions_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
mkdir "${RUN}/actions"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for path in "${STATES}" "${POLICY}/adapter_model.safetensors"; do [[ -f "${path}" ]]; done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 2 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" "${TORCHRUN}" --standalone --nproc_per_node=2 \
  "${SOURCE}/src/scripts/build_spad_basin_preflight_actions.py" \
  --states-jsonl "${STATES}" --model-path "${MODEL}" \
  --checkpoint-path "${POLICY}" --crysllmgen-dir "${CRYS}" \
  --output-dir "${RUN}/actions" --chgnet-batch-size 16 --original-batch-size 8 \
  >"${RUN}/logs/actions.out" 2>"${RUN}/logs/actions.err"

"${PYTHON}" - "${RUN}" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
groups = []
rank_reports = []
for rank in range(2):
    rank_reports.append(json.loads((run / "actions" / f"report_rank{rank}.json").read_text()))
    with (run / "actions" / f"groups_rank{rank}.jsonl").open() as handle:
        groups.extend(json.loads(line) for line in handle if line.strip())
groups.sort(key=lambda row: int(row["sample_idx"]))
if len(groups) != 128 or [int(row["sample_idx"]) for row in groups] != list(range(128)):
    raise RuntimeError("distributed action merge changed the fixed 128-state denominator")

expected_sources = ["no_op", "reference_dlm", "physics_downhill", "physics_reverse"]
contract_errors = []
k_hist = Counter()
attempt_status = Counter()
proposal_status = Counter()
terminal_valid = 0
reference_retained = 0
reference_matched = 0
reference_mismatch = 0
with (run / "groups.jsonl").open("x", encoding="utf-8", newline="\n") as output:
    for group in groups:
        attempts = group.get("candidate_attempts") or []
        candidates = group.get("candidates") or []
        if [row.get("source") for row in attempts] != expected_sources:
            contract_errors.append(f"attempt_sources:{group['sample_idx']}")
        if not 1 <= len(candidates) <= 4:
            contract_errors.append(f"candidate_count:{group['sample_idx']}:{len(candidates)}")
        if group.get("outcomes_read") is not False or group.get("selection") is not False or group.get("replacement") is not False:
            contract_errors.append(f"outcome_contract:{group['sample_idx']}")
        k_hist[len(candidates)] += 1
        attempt_status.update(str(row.get("retention_status")) for row in attempts)
        proposal_status.update(str(row.get("proposal_status")) for row in attempts)
        terminal_valid += sum(row.get("terminal_legal") is True for row in candidates)
        replay = group.get("reference_replay") or {}
        if replay.get("retained") is True:
            reference_retained += 1
            if replay.get("matches_final") is True:
                reference_matched += 1
            else:
                reference_mismatch += 1
        output.write(json.dumps(group, ensure_ascii=False, sort_keys=True) + "\n")

report = {
    "schema": "spad_basin_preflight_actions_final_v1",
    "groups": len(groups),
    "world_size": 2,
    "candidate_count_histogram": {str(k): int(v) for k, v in sorted(k_hist.items())},
    "candidate_attempt_status": dict(sorted(attempt_status.items())),
    "proposal_status": dict(sorted(proposal_status.items())),
    "retained_candidates": sum(len(row.get("candidates") or []) for row in groups),
    "terminal_valid_candidates": terminal_valid,
    "reference_replay": {
        "retained": reference_retained,
        "matched": reference_matched,
        "mismatched": reference_mismatch,
    },
    "contract_errors": contract_errors,
    "outcomes_read": False,
    "selection": False,
    "replacement": False,
    "rank_reports": rank_reports,
}
(run / "ACTIONS_FINAL.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if reference_mismatch or contract_errors:
    raise RuntimeError(
        f"action build failed: reference_mismatch={reference_mismatch} contract_errors={len(contract_errors)}"
    )
PY

trap - ERR
touch "${RUN}/_SUCCESS"

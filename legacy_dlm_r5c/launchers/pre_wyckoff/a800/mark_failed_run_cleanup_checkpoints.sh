#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
LOWLR5_CHECKPOINT="${LOWLR5_CHECKPOINT:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
SFTBEST_CHECKPOINT="${SFTBEST_CHECKPOINT:-runs/20260521_211500-final07-refined-seal1/outputs/sft_refined_seal1/final}"

cd "${PROJECT_ROOT}"

RUN_DIR="runs/${RUN_ID}"
NOTES_DIR="${RUN_DIR}/notes"
mkdir -p "${NOTES_DIR}"
export RUN_ID PROJECT_ROOT LOWLR5_CHECKPOINT SFTBEST_CHECKPOINT

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

run_id = os.environ["RUN_ID"]
project_root = Path(os.environ.get("PROJECT_ROOT", "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion")).resolve()
run_dir = project_root / "runs" / run_id
notes_dir = run_dir / "notes"
notes_dir.mkdir(parents=True, exist_ok=True)

keep_paths = {
    (project_root / os.environ["LOWLR5_CHECKPOINT"]).resolve(),
    (project_root / os.environ["SFTBEST_CHECKPOINT"]).resolve(),
}


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


main_gate = read_json(notes_dir / "dynamic_stage_a_final_gate256.json")
main_comp = read_json(notes_dir / "dynamic_stage_a_final_composition256.json")
nprior_gate = read_json(notes_dir / "dynamic_stage_a_final_trainNprior_gate256.json")
nprior_comp = read_json(notes_dir / "dynamic_stage_a_final_trainNprior_composition256.json")
decision = read_json(notes_dir / "dynamic_final_decision.json")


def metrics_from_gate(gate):
    return dict((gate or {}).get("metrics") or {})


def reason_counts(comp):
    raw = (comp or {}).get("raw_jsonl") or (comp or {}).get("raw_pt") or {}
    return dict(raw.get("reason_counts") or {})


def hist(comp, key):
    raw = (comp or {}).get("raw_jsonl") or (comp or {}).get("raw_pt") or {}
    return dict(raw.get(key) or {})


failure_summary = {
    "run_id": run_id,
    "status": "failed",
    "failure_type": "dynamic_v1_conservative_trial_failed_256_gate",
    "official_decision": (decision or {}).get("decision", "stop_after_256"),
    "official_reason": (decision or {}).get("reason", "best candidate did not reach expansion gate"),
    "main_256": {
        "metrics": metrics_from_gate(main_gate),
        "reason_counts": reason_counts(main_comp),
        "num_atoms_histogram": hist(main_comp, "num_atoms_histogram"),
        "num_elements_histogram": hist(main_comp, "num_elements_histogram"),
    },
    "diagnostic_train_n_prior_256": {
        "metrics": metrics_from_gate(nprior_gate),
        "reason_counts": reason_counts(nprior_comp),
        "num_atoms_histogram": hist(nprior_comp, "num_atoms_histogram"),
        "num_elements_histogram": hist(nprior_comp, "num_elements_histogram"),
        "note": "diagnostic only; not accepted because comp_valid increase is dominated by shortcuts",
    },
    "accepted_for_1000": False,
    "cleanup": {
        "keep_paths": [str(path) for path in sorted(keep_paths)],
    },
}
(notes_dir / "dynamic_v1_failure_summary.json").write_text(
    json.dumps(failure_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

main_m = failure_summary["main_256"]["metrics"]
nprior_m = failure_summary["diagnostic_train_n_prior_256"]["metrics"]
main_reasons = failure_summary["main_256"]["reason_counts"]
nprior_reasons = failure_summary["diagnostic_train_n_prior_256"]["reason_counts"]
main_n_hist = failure_summary["main_256"]["num_atoms_histogram"]

lines = [
    "# Dynamic-v1 Conservative Trial Failure",
    "",
    f"- Run: `{run_id}`",
    "- Status: failed; stopped after 256 smoke, no 1000/refinement/S.U.N. expansion.",
    "- Preserved: logs, notes, raw generations, failure cases, sample metrics.",
    "- Cleanup: deleting only checkpoint/final directories that contain model weight files.",
    "",
    "## Main 256 Smoke",
    "",
    f"- parse: {main_m.get('parse_rate', 0):.4f}",
    f"- graph_acceptance: {main_m.get('graph_acceptance', 0):.4f}",
    f"- comp_valid: {main_m.get('comp_valid', 0):.4f}",
    f"- strict_valid: {main_m.get('strict_valid', 0):.4f}",
    f"- shortcut: {main_m.get('shortcut', 0):.4f}",
    f"- pbc_duplicate: {main_m.get('pbc_duplicate', 0):.4f}",
    f"- reason_counts: `{json.dumps(main_reasons, ensure_ascii=False, sort_keys=True)}`",
    f"- num_atoms_histogram: `{json.dumps(main_n_hist, ensure_ascii=False, sort_keys=True)}`",
    "",
    "Main failure diagnosis: after fixing the dynamic duplicate mask, graph validity recovered to 1.0, but raw comp_valid stayed below the expansion threshold. The remaining failures are charge-neutrality failures, and N collapsed toward 18/20 atom outputs.",
    "",
    "## Train N-Prior Diagnostic",
    "",
    f"- parse: {nprior_m.get('parse_rate', 0):.4f}",
    f"- graph_acceptance: {nprior_m.get('graph_acceptance', 0):.4f}",
    f"- comp_valid: {nprior_m.get('comp_valid', 0):.4f}",
    f"- strict_valid: {nprior_m.get('strict_valid', 0):.4f}",
    f"- shortcut: {nprior_m.get('shortcut', 0):.4f}",
    f"- single_element: {nprior_m.get('single_element', 0):.4f}",
    f"- all_metal: {nprior_m.get('all_metal', 0):.4f}",
    f"- reason_counts: `{json.dumps(nprior_reasons, ensure_ascii=False, sort_keys=True)}`",
    "",
    "N-prior diagnosis: restoring the MP-20 atom-count distribution raises raw comp_valid, but mostly by shortcut behavior, especially single-element shortcut, so this is not an acceptable improvement.",
    "",
]
(notes_dir / "dynamic_v1_failure_report.md").write_text("\n".join(lines), encoding="utf-8")

weight_names = {
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
}
candidate_dirs = set()
for path in run_dir.rglob("*"):
    if not path.is_file():
        continue
    if path.name in weight_names or (path.name.startswith("pytorch_model-") and path.name.endswith(".bin")):
        candidate_dirs.add(path.parent.resolve())

deleted = []
skipped = []
for directory in sorted(candidate_dirs):
    if any(directory == keep or keep in directory.parents for keep in keep_paths):
        skipped.append({"path": str(directory), "reason": "preserved_checkpoint"})
        continue
    if not directory.exists():
        skipped.append({"path": str(directory), "reason": "missing"})
        continue
    rel = directory.relative_to(project_root)
    if not str(rel).startswith(f"runs/{run_id}/"):
        skipped.append({"path": str(directory), "reason": "outside_run_dir"})
        continue
    size_bytes = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    shutil.rmtree(directory)
    deleted.append({"path": str(rel), "size_bytes": int(size_bytes)})

manifest = {
    "run_id": run_id,
    "deleted_count": len(deleted),
    "deleted_size_bytes": sum(item["size_bytes"] for item in deleted),
    "deleted": deleted,
    "skipped": skipped,
    "failure_summary": str((notes_dir / "dynamic_v1_failure_summary.json").relative_to(project_root)),
    "failure_report": str((notes_dir / "dynamic_v1_failure_report.md").relative_to(project_root)),
}
(notes_dir / "failed_checkpoint_cleanup_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

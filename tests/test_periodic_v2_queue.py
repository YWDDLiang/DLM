from copy import deepcopy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("v2_queue", ROOT / "scripts/check_periodic_v2_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def complete_predecessors(tmp_path):
    manifest = json.loads((ROOT / "docs/periodic_self_repair_v1/V2_LAUNCH_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update(ready=True, implementation_commit="a" * 40)
    for ordinal, entry in enumerate(manifest["prerequisites"]):
        if "job_pointer" in entry:
            pointer = tmp_path / entry["job_pointer"]
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(str(50000 + ordinal))
            run = tmp_path / entry["run_pattern"].format(job_id=50000 + ordinal)
        else:
            run = tmp_path / entry["run_dir"]
        for marker in entry["success_markers"]:
            path = run / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for item in entry.get("reports", []):
            report = {}
            for dotted, value in item["equals"].items():
                current = report
                parts = dotted.split(".")
                for key in parts[:-1]:
                    current = current.setdefault(key, {})
                current[parts[-1]] = value
            path = run / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))
    return manifest


def test_empty_scheduler_cannot_bypass_unsubmitted_later_phase(tmp_path):
    manifest = complete_predecessors(tmp_path)
    assert queue.assess_queue(tmp_path, manifest)["ready"]
    (tmp_path / "runs/K8_MAIN_EVALUATION_JOB").unlink()
    report = queue.assess_queue(tmp_path, manifest, active_jobs=[])
    assert not report["ready"]
    assert "predecessor_incomplete:old_k8_main_evaluation" in report["reasons"]


def test_success_marker_does_not_hide_wrong_evaluation_denominator(tmp_path):
    manifest = complete_predecessors(tmp_path)
    entry = next(x for x in manifest["prerequisites"] if x["id"] == "old_k8_tau800")
    report_path = tmp_path / entry["run_dir"] / "evaluation/EVALUATION_FINAL.json"
    report = json.loads(report_path.read_text())
    report["counts"]["requests"] = 252
    report_path.write_text(json.dumps(report))
    assert not queue.assess_queue(tmp_path, manifest)["ready"]


def test_only_own_allocation_can_be_excluded_and_other_project_is_not_managed(tmp_path):
    manifest = complete_predecessors(tmp_path)
    jobs = queue.project_jobs("60000|periodic-v2-train|RUNNING\n60001|force-resid|RUNNING\n", allowed_job_id="60000")
    assert jobs == []
    assert queue.assess_queue(tmp_path, manifest, active_jobs=jobs)["ready"]
    jobs = queue.project_jobs("60000|periodic-v2-train|PENDING\n60001|spad-state-main|RUNNING\n")
    assert not queue.assess_queue(tmp_path, manifest, active_jobs=jobs)["ready"]
    unfinished = deepcopy(manifest)
    unfinished["ready"] = False
    assert not queue.assess_queue(tmp_path, unfinished)["ready"]

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("supplement_gate", Path(__file__).parents[1] /
                                            "operations/supplement_1000/check_before_v2.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def registry(tmp_path, rule="final_sun_goal"):
    data = {"schema": "qualifying_policy_1000_registry_v1", "qualification_rule": rule, "candidates": {}}
    for name in ("original_k4", "original_k8", "raw_v1", "v2"):
        item = {"before_v2": name != "v2", "development_evaluations": {},
                "policy_path": str(tmp_path / name / "policy"), "supplement_run": None}
        for endpoint in ("native", "tau800"):
            directory = tmp_path / name / endpoint
            write(directory / "EVALUATION_FINAL.json", {"endpoint": endpoint,
                  "cohort_role": "fixed_development", "counts": {
                      "requests": 256, "reconstructed": 254, "strict_sun": 7, "meta_sun": 57}})
            (directory / "_SUCCESS").touch()
            item["development_evaluations"][endpoint] = str(directory)
        data["candidates"][name] = item
    path = tmp_path / "REGISTRY.json"
    write(path, data)
    return path, data


def test_final_goal_does_not_mistake_old_posttraining_admission_for_target(tmp_path):
    path, data = registry(tmp_path)
    assert gate.inspect_registry(path)["ready"]
    data["qualification_rule"] = "posttrain_admission"
    write(path, data)
    report = gate.inspect_registry(path)
    assert not report["ready"] and report["candidates"]["original_k4"]["admitted"]


def test_one_endpoint_reaching_final_target_triggers_supplement(tmp_path):
    path, data = registry(tmp_path)
    directory = Path(data["candidates"]["raw_v1"]["development_evaluations"]["native"])
    report = json.loads((directory / "EVALUATION_FINAL.json").read_text())
    report["counts"].update(strict_sun=26, meta_sun=128)
    write(directory / "EVALUATION_FINAL.json", report)
    result = gate.inspect_registry(path)
    assert result["candidates"]["raw_v1"]["endpoint_pass"] == {"native": True, "tau800": False}
    assert not result["ready"]


def test_full1200_success_does_not_replace_matched1000_completion(tmp_path):
    path, data = registry(tmp_path)
    item = data["candidates"]["raw_v1"]
    native = Path(item["development_evaluations"]["native"])
    dev = json.loads((native / "EVALUATION_FINAL.json").read_text())
    dev["counts"].update(strict_sun=26, meta_sun=128)
    write(native / "EVALUATION_FINAL.json", dev)
    run = tmp_path / "supplement"
    item["supplement_run"] = str(run)
    write(path, data)
    selection = {"selected_source_ordinals": list(range(1000)), "selected_parseable_cifs": 1000,
                 "selection_basis": "CIF_parser_only_in_source_order", "energy_or_stability_selection": False}
    write(run / "selection/SELECTION_FINAL.json", selection)
    write(run / "METHOD_LOCK.json", {"policy_path": item["policy_path"]})
    write(run / "sample/SAMPLE_FINAL.json", {"checkpoint": item["policy_path"]})
    for endpoint in ("native", "tau800"):
        write(run / f"{endpoint}-evaluation/EVALUATION_FINAL.json", {"endpoint": endpoint,
              "counts": {"requests": 1200}, "cohort_role": "independent_main",
              "conditional_1000": {"counts": {"requests": 1000}, "selection": selection}})
    for marker in ("_SUCCESS", "selection/_SUCCESS", "native-evaluation/_SUCCESS", "tau800-evaluation/_SUCCESS"):
        (run / marker).touch()
    assert gate.inspect_registry(path)["ready"]
    tau = run / "tau800-evaluation/EVALUATION_FINAL.json"
    report = json.loads(tau.read_text())
    report["conditional_1000"]["selection"]["selected_source_ordinals"][-1] = 1001
    write(tau, report)
    assert not gate.inspect_registry(path)["ready"]

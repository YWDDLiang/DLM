#!/usr/bin/env python3
"""Additional user-authorized 1000-sample work must finish before V2 starts."""
import argparse
import json
from pathlib import Path

RULES = {
    "final_sun_goal": {
        "native": {"strict_sun": 26, "meta_sun": 128},
        "tau800": {"strict_sun": 26, "meta_sun": 128},
    },
    "posttrain_admission": {
        "native": {"reconstructed": 252, "strict_sun": 6, "meta_sun": 55},
        "tau800": {"reconstructed": 252, "strict_sun": 18, "meta_sun": 124},
    },
}


def inspect_registry(path):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema") != "qualifying_policy_1000_registry_v1":
        raise ValueError("unknown supplement registry")
    candidates = registry["candidates"]
    thresholds_by_endpoint = RULES[registry["qualification_rule"]]
    if not {"original_k4", "original_k8", "raw_v1", "v2"} <= set(candidates):
        raise ValueError("a registered candidate is missing")
    result, reasons = {}, []
    for name, item in candidates.items():
        if not item["before_v2"]:
            continue
        problems, passes = [], {}
        for endpoint, thresholds in thresholds_by_endpoint.items():
            try:
                directory = Path(item["development_evaluations"][endpoint])
                assert (directory / "_SUCCESS").is_file(), "development endpoint incomplete"
                report = json.loads((directory / "EVALUATION_FINAL.json").read_text(encoding="utf-8"))
                assert report["endpoint"] == endpoint and report["cohort_role"] == "fixed_development"
                assert report["counts"]["requests"] == 256, "wrong development denominator"
                passes[endpoint] = all(report["counts"][key] >= value for key, value in thresholds.items())
            except (OSError, KeyError, TypeError, ValueError, AssertionError) as error:
                problems.append(f"{endpoint}: {error}")
        admitted = any(passes.values())
        if admitted:
            try:
                run = Path(item["supplement_run"])
                for marker in ("_SUCCESS", "selection/_SUCCESS", "native-evaluation/_SUCCESS", "tau800-evaluation/_SUCCESS"):
                    assert (run / marker).is_file(), "supplement incomplete: " + marker
                selection = json.loads((run / "selection/SELECTION_FINAL.json").read_text(encoding="utf-8"))
                selected = selection["selected_source_ordinals"]
                assert len(selected) == 1000 and selected == sorted(set(selected))
                assert selection["selected_parseable_cifs"] == 1000
                assert selection["selection_basis"] == "CIF_parser_only_in_source_order"
                assert selection["energy_or_stability_selection"] is False
                for endpoint, directory in (("native", "native-evaluation"), ("tau800", "tau800-evaluation")):
                    report = json.loads((run / directory / "EVALUATION_FINAL.json").read_text(encoding="utf-8"))
                    assert report["endpoint"] == endpoint and report["counts"]["requests"] == 1200
                    assert report["cohort_role"] == "independent_main"
                    assert report["conditional_1000"]["counts"]["requests"] == 1000
                    assert report["conditional_1000"]["selection"]["selected_source_ordinals"] == selected
                lock_path = Path(item["method_lock"]) if item.get("method_lock") else run / "METHOD_LOCK.json"
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                assert Path(lock["policy_path"]).resolve() == Path(item["policy_path"]).resolve()
                sample = json.loads((run / "sample/SAMPLE_FINAL.json").read_text(encoding="utf-8"))
                assert Path(sample["checkpoint"]).resolve() == Path(item["policy_path"]).resolve()
            except (OSError, KeyError, TypeError, ValueError, AssertionError) as error:
                problems.append(str(error))
        result[name] = {"admitted": admitted, "endpoint_pass": passes, "complete": not problems, "problems": problems}
        if problems:
            reasons.append(name)
    return {"ready": not reasons, "incomplete_candidates": reasons, "candidates": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_registry(args.registry)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()

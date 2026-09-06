import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("cohort_hull_coverage", Path(__file__).parents[1] /
                                            "operations/supplement_1000/check_hull_coverage.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_completed_cache_for_old_cohort_cannot_hide_new_chemistries():
    plans = [{"plan_state": {"elements": ["O", "Fe"]}}, {"plan_state": {"elements": ["Li", "F"]}}]
    report = module.coverage(plans, [{"chemsys": "Fe-O"}], [])
    assert not report["coverage_accounted"] and report["unqueried_chemsys"] == ["F-Li"]


def test_explicit_query_failure_is_accounted_but_remains_unknown():
    plans = [{"plan_state": {"elements": ["Li", "F"]}}]
    report = module.coverage(plans, [], [{"chemsys": "F-Li"}])
    assert report["coverage_accounted"] and report["resolved_chemsys"] == 0
    assert report["explicitly_unresolved_chemsys"] == ["F-Li"]
    assert report["unknown_is_not_zero_energy"]
    with pytest.raises(ValueError):
        module.coverage(plans, [{"chemsys": "F-Li"}], [{"chemsys": "F-Li"}])

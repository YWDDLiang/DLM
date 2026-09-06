from copy import deepcopy
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("construction_endpoint", Path(__file__).parents[1] /
                                            "scripts/extract_periodic_construction_endpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_repair_rollback_does_not_overwrite_the_measured_construction_prefix():
    record = {"sample_idx": 0, "source_split": "evaluation", "trace": {
        "initial_body": [1, 99, 99], "mask_id": 99, "success": True, "events": [
            {"op": "begin", "phase": "construct", "positions": [1, 2]},
            {"op": "draw", "phase": "construct", "position": 1, "token": 4},
            {"op": "draw", "phase": "construct", "position": 2, "token": 5},
            {"op": "end", "phase": "construct"},
            {"op": "begin", "phase": "full_cell_repair", "positions": [1, 2]},
            {"op": "draw", "phase": "full_cell_repair", "position": 1, "token": 8},
            {"op": "rollback", "phase": "full_cell_repair", "positions": [1, 2], "reason": "empty_support"},
            {"op": "end", "phase": "full_cell_repair"}]}}
    untouched = deepcopy(record)
    out = module.construction_endpoint(record, lambda values: str(values))
    assert out["final_body_token_ids"] == [1, 4, 5]
    assert out["success"] and out["repair_transaction_entered"]
    assert len(out["trace"]["events"]) == 4
    assert record == untouched


def test_failed_construction_is_retained_without_replacement_or_success_relabel():
    record = {"sample_idx": 3, "source_split": "evaluation", "trace": {
        "initial_body": [1, 99], "mask_id": 99, "success": False,
        "failure": "construction_empty_support", "events": []}}
    out = module.construction_endpoint(record, lambda values: str(values))
    assert out["sample_idx"] == 3 and not out["success"]
    assert out["final_body_token_ids"] == [1, 99] and not out["repair_transaction_entered"]

from __future__ import annotations

import ast
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "crystal_dlm"
    / "wqcodiff"
    / "recovery.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing top-level function: {name}")


def _called_names(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class RecoveryFunctionBoundaryTests(unittest.TestCase):
    def test_reverse_update_and_cpu_prepare_remain_separate_top_level_paths(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        advance = _function(tree, "_advance_recovery_work")
        prepare = _function(tree, "_prepare_recovery_work")
        batch = _function(tree, "_run_recovery_batch")

        advance_calls = _called_names(advance)
        self.assertTrue(
            {
                "_continuous_step",
                "_replace_masked_fields",
                "_d3pm_reverse_fields",
                "_event_logits",
                "_apply_event",
            }
            <= advance_calls
        )
        self.assertTrue(
            {"expand_state", "compute_geometry_evidence", "tensorize_state"}
            <= _called_names(prepare)
        )
        self.assertTrue(
            {"_prepare_recovery_work", "_advance_recovery_work"}
            <= _called_names(batch)
        )
        self.assertNotIn("_prepare_recovery_work", advance_calls)


if __name__ == "__main__":
    unittest.main()

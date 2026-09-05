"""Keep retained constraint-constructor contracts checked without loading HF."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StateEntrypointContractTest(unittest.TestCase):
    def test_constraint_calls_supply_all_required_keyword_arguments(self):
        source = ast.parse((ROOT / "src/scripts/sample_llada_dynamic_crystals.py").read_text(encoding="utf-8"))
        definition = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                          and node.name == "build_dynamic_lightweight_constraints")
        required = {arg.arg for arg, default in zip(definition.args.kwonlyargs, definition.args.kw_defaults)
                    if default is None}
        for filename in ("preflight_state_programmed_spad.py", "train_state_conditioned_spad.py", "sample_state_programmed_paths.py", "train_programmed_path_policy.py"):
            tree = ast.parse((ROOT / "src/scripts" / filename).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == definition.name]
            self.assertTrue(calls, filename)
            for call in calls:
                supplied = {item.arg for item in call.keywords}
                self.assertFalse(required - supplied, (filename, required - supplied))


if __name__ == "__main__":
    unittest.main()

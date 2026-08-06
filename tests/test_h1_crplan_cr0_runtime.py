from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from crystal_dlm.h1_crplan import CRPlanTokenVocabulary, OxidationReachability


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "workstreams"
    / "plangraph_dlm_iclr_20260731"
    / "execution"
    / "h1_crplan_r0_paired32_script_package_repair_v5"
    / "run_cr0_audit.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "h1_crplan_script_package_repair_v5_audit",
        AUDIT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audit module from {AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __len__(self) -> int:
        return 5


class CR0RuntimeRepairTests(unittest.TestCase):
    def test_repository_scripts_package_marker_is_frozen(self) -> None:
        marker = ROOT / "scripts" / "__init__.py"
        self.assertTrue(marker.is_file())
        self.assertIn(
            "prevents unrelated site-packages",
            marker.read_text(encoding="utf-8"),
        )
        spec = importlib.util.find_spec(
            "scripts.sample_llada_dynamic_crystals"
        )
        self.assertIsNotNone(spec)
        self.assertEqual(
            Path(spec.origin).resolve(),
            (ROOT / "scripts" / "sample_llada_dynamic_crystals.py").resolve(),
        )

    def test_direct_evaluator_counts_match_frozen_gcd_reduction(self) -> None:
        module = load_audit_module()
        self.assertEqual(module.direct_evaluator_counts((2, 2)), [1, 1])
        self.assertEqual(module.direct_evaluator_counts((2, 4)), [1, 2])
        self.assertEqual(module.direct_evaluator_counts((1, 3)), [1, 3])
        with self.assertRaises(ValueError):
            module.direct_evaluator_counts((2, 0))

    def test_backend_probe_disable_is_cr0_only(self) -> None:
        execution_dir = AUDIT_PATH.parent
        cr0 = (execution_dir / "cr0.sbatch").read_text(encoding="utf-8")
        paired32 = (execution_dir / "paired32.sbatch").read_text(
            encoding="utf-8"
        )
        for name in ("USE_TORCH", "USE_TF", "USE_FLAX", "USE_TORCH_XLA"):
            self.assertIn(f"export {name}=0", cr0)
            self.assertNotIn(f"export {name}=0", paired32)

    def test_probability_audit_is_full_vocab_and_torch_free(self) -> None:
        module = load_audit_module()
        reachability = OxidationReachability(
            {"Fe": (2, 3), "O": (-2,)},
            metals=("Fe",),
            max_atoms=20,
            table_source="unit_test",
            table_version="1",
        )
        vocabulary = CRPlanTokenVocabulary(
            ("a", "b", "c", "d", "e"),
            eos_token_id=4,
        )
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("CR-0 probability audit imported torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            report = module.probability_and_empty_support_audit(
                FakeTokenizer(),
                vocabulary,
                reachability,
            )

        self.assertTrue(report["after_formula_support_is_full_vocab"])
        self.assertTrue(report["same_full_support_probability_parity"])
        self.assertEqual(report["maximum_absolute_probability_difference"], 0)
        self.assertTrue(report["empty_support_detected"])
        self.assertFalse(report["explicit_torch_import_used"])


if __name__ == "__main__":
    unittest.main()

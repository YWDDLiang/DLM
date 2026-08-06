from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.atom_text import (
    parse_upstream_atom_text_fields,
)
from crystal_dlm.wqcodiff.crysllmgen.parity import (
    REQUIRED_CHECKS,
    ParityContract,
    audit_parity_report,
    compare_values,
    hash_fixed_select,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "crysllmgen_parity_v1.json"
)


class CrysLLMGenAtomTextTests(unittest.TestCase):
    def test_parser_preserves_upstream_nonempty_line_semantics(self) -> None:
        fields = parse_upstream_atom_text_fields(
            "5.1 5.2 5.3\n90 91 92\n\nSi\n0.1 0.2 0.3\nO\n0.4 0.5 0.6\n"
        )
        self.assertEqual(fields.lengths, (5.1, 5.2, 5.3))
        self.assertEqual(fields.angles, (90.0, 91.0, 92.0))
        self.assertEqual(fields.species, ("Si", "O"))
        self.assertEqual(fields.num_atoms, 2)

    def test_parser_does_not_repair_odd_or_prefixed_text(self) -> None:
        with self.assertRaises(ValueError):
            parse_upstream_atom_text_fields("answer\n5 5 5\n90 90 90\nSi\n0 0 0")
        with self.assertRaises(ValueError):
            parse_upstream_atom_text_fields("5 5 5\n90 90 90\nSi")


class CrysLLMGenParityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ParityContract.load(CONTRACT_PATH)

    def test_contract_is_strict_and_complete(self) -> None:
        self.assertEqual(self.contract.proposal_count, 256)
        self.assertEqual(self.contract.required_checks, REQUIRED_CHECKS)
        self.assertLessEqual(self.contract.absolute_tolerance, 1.0e-6)
        self.assertEqual(self.contract.relative_tolerance, 0.0)
        self.assertEqual(self.contract.scheduler_timesteps, 1000)
        self.assertEqual(self.contract.parent_run_type, "train")
        self.assertEqual(self.contract.one_step_device, "cpu")

    def test_hash_fixed_selection_is_input_order_independent(self) -> None:
        records = [{"id": f"mp-{index}"} for index in range(1000)]
        first = hash_fixed_select(
            records,
            identity=lambda row: row["id"],
            count=256,
            salt=self.contract.selection_salt,
        )
        shuffled = list(records)
        random.Random(19).shuffle(shuffled)
        second = hash_fixed_select(
            shuffled,
            identity=lambda row: row["id"],
            count=256,
            salt=self.contract.selection_salt,
        )
        self.assertEqual(first, second)

    def test_nested_numeric_comparison_reports_first_failure(self) -> None:
        passed = compare_values(
            {"x": [1.0, 2.0], "label": "same"},
            {"x": [1.0 + 5.0e-7, 2.0], "label": "same"},
            absolute_tolerance=1.0e-6,
            relative_tolerance=0.0,
        )
        self.assertTrue(passed.passed)
        failed = compare_values(
            [1.0, 2.0],
            [1.0, 2.0 + 2.0e-6],
            absolute_tolerance=1.0e-6,
            relative_tolerance=0.0,
        )
        self.assertFalse(failed.passed)
        self.assertIn("root[1]", failed.first_mismatch or "")

    def test_nested_tensor_like_values_are_hashable(self) -> None:
        class TensorLike:
            def __init__(self, values: list[float]) -> None:
                self.values = values

            def detach(self) -> "TensorLike":
                return self

            def cpu(self) -> "TensorLike":
                return self

            def tolist(self) -> list[float]:
                return self.values

        comparison = compare_values(
            {"schedule": TensorLike([0.1, 0.2, 0.3])},
            {"schedule": [0.1, 0.2, 0.3]},
            absolute_tolerance=1.0e-6,
            relative_tolerance=0.0,
        )
        self.assertTrue(comparison.passed)
        self.assertEqual(comparison.numeric_values, 3)
        self.assertEqual(len(comparison.upstream_sha256), 64)

    def test_auditor_rejects_partial_or_retried_report(self) -> None:
        checks = {
            name: {"passed": True, "max_absolute_error": 0.0}
            for name in REQUIRED_CHECKS
        }
        report = {
            "schema": "crysllmgen_disabled_extension_parity_report_v1",
            "contract_sha256": self.contract.sha256,
            "proposal_count": 256,
            "terminal_attempts": 256,
            "retry_or_replacement_used": False,
            "checks": checks,
        }
        self.assertTrue(audit_parity_report(report, self.contract)["ok"])
        report["retry_or_replacement_used"] = True
        self.assertFalse(audit_parity_report(report, self.contract)["ok"])

    def test_report_writer_is_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_exclusive(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()

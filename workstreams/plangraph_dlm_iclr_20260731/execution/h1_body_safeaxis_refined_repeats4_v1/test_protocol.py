from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemble_terminal import ENDPOINTS, _hierarchical_bootstrap, _mcnemar
from protocol import (
    DENOMINATOR,
    attempt_id,
    ordered_rows,
    read_json,
    validate_arm,
    validate_config,
    validate_repeat,
)


HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_frozen_config(self) -> None:
        config = read_json(HERE / "config.json")
        validate_config(config)
        self.assertEqual(
            config["protocol"]["arm_order"],
            [
                ["control", "candidate"],
                ["candidate", "control"],
                ["candidate", "control"],
                ["control", "candidate"],
            ],
        )

    def test_arm_repeat_attempt_identity(self) -> None:
        self.assertEqual(validate_arm("control"), "control")
        self.assertEqual(validate_repeat("3"), 3)
        self.assertEqual(
            attempt_id(2, "candidate", 7),
            "h1-r03e-r2-candidate-0007",
        )
        with self.assertRaises(ValueError):
            validate_arm("M00")
        with self.assertRaises(ValueError):
            validate_repeat(4)

    def test_ordered_rows_fails_closed(self) -> None:
        rows = [{"ordinal": index} for index in reversed(range(DENOMINATOR))]
        ordered = ordered_rows(rows, ordinal_field="ordinal")
        self.assertEqual([row["ordinal"] for row in ordered], list(range(256)))
        with self.assertRaises(ValueError):
            ordered_rows(rows[:-1], ordinal_field="ordinal")

    def test_exact_mcnemar(self) -> None:
        control = np.asarray([True, True, False, False], dtype=bool)
        candidate = np.asarray([False, True, True, False], dtype=bool)
        result = _mcnemar(control, candidate)
        self.assertEqual(result["control_only"], 1)
        self.assertEqual(result["candidate_only"], 1)
        self.assertEqual(result["exact_two_sided_p"], 1.0)

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        differences = np.zeros((4, 256, len(ENDPOINTS)), dtype=float)
        differences[:, :, 0] = 1.0
        first = _hierarchical_bootstrap(differences, seed=17, replicates=1000)
        second = _hierarchical_bootstrap(differences, seed=17, replicates=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["generation_complete"]["mean_delta"], 1.0)
        self.assertEqual(
            first["generation_complete"][
                "hierarchical_paired_bootstrap_95ci"
            ],
            [1.0, 1.0],
        )

    def test_json_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)


if __name__ == "__main__":
    unittest.main()

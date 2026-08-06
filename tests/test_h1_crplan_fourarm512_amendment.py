from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.evaluate_fourarm512 import (
    DENOMINATOR,
    MODES,
    exact_mcnemar_p,
    scientific_gates_from_metrics,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.generate_science_ledger import (
    ATTEMPTS,
    BASE_SEED,
    build_ledger,
)
from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_fourarm512_route_amendment_v1.run_fourarm512_arm import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = (
    ROOT
    / "workstreams"
    / "plangraph_dlm_iclr_20260731"
    / "execution"
    / "h1_crplan_fourarm512_route_amendment_v1"
)


def metric_fixture(
    *,
    nonshortcut: int,
    composition: int,
    primary: int,
    parsed: int = 500,
    completion: int = 512,
    shortcut: int = 5,
    unique_rate: float = 0.90,
    coverage_rate: float = 0.70,
    applicable: int = 400,
    affected: int = 40,
) -> dict[str, object]:
    return {
        "nonshortcut_composition_valid_count": nonshortcut,
        "composition_valid_count": composition,
        "primary_composition_valid_count": primary,
        "parse_count": parsed,
        "completion_count": completion,
        "shortcut_valid_count": shortcut,
        "unique_formula_rate": unique_rate,
        "element_coverage_rate": coverage_rate,
        "charge_applicable_count": applicable,
        "charge_applicable_terminal_failure_count": 0,
        "mechanism_telemetry": {
            "preterminal_affected_attempt_count": affected,
        },
    }


class FourArm512ContractTests(unittest.TestCase):
    def test_science_ledger_is_exact_and_frozen(self) -> None:
        ledger_path = EXECUTION / "SCIENCE_LEDGER.json"
        config = json.loads(
            (EXECUTION / "CONFIG.json").read_text(encoding="utf-8")
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger, build_ledger())
        self.assertEqual(ATTEMPTS, DENOMINATOR)
        self.assertEqual(BASE_SEED, 1187798901)
        self.assertEqual(
            hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            config["science_ledger_sha256"],
        )
        self.assertEqual(
            [row["ordinal"] for row in ledger["rows"]],
            list(range(DENOMINATOR)),
        )
        self.assertEqual(
            len({row["planner_sampling_seed"] for row in ledger["rows"]}),
            DENOMINATOR,
        )
        for mode in MODES:
            seeds = validate_contract(
                config=config,
                ledger=ledger,
                ledger_sha256=config["science_ledger_sha256"],
                mode=mode,
            )
            self.assertEqual(len(seeds), DENOMINATOR)

    def test_primary_gate_passes_exactly_at_registered_boundary(self) -> None:
        terminal = metric_fixture(
            nonshortcut=300,
            composition=305,
            primary=280,
        )
        full = metric_fixture(
            nonshortcut=311,
            composition=316,
            primary=290,
            affected=20,
        )
        gates, deltas = scientific_gates_from_metrics(
            terminal=terminal,
            full=full,
        )
        self.assertTrue(all(gates.values()))
        self.assertEqual(
            deltas["nonshortcut_composition_valid_count"],
            11,
        )
        self.assertEqual(
            deltas["charge_applicable_preterminal_affected_rate"],
            0.05,
        )

    def test_shortcut_gain_does_not_satisfy_primary_gate(self) -> None:
        terminal = metric_fixture(
            nonshortcut=300,
            composition=305,
            primary=280,
            shortcut=5,
        )
        full = metric_fixture(
            nonshortcut=305,
            composition=320,
            primary=285,
            shortcut=15,
        )
        gates, _ = scientific_gates_from_metrics(
            terminal=terminal,
            full=full,
        )
        self.assertFalse(
            gates["nonshortcut_composition_valid_gain_ge_11"]
        )
        self.assertFalse(
            gates["full_shortcut_valid_count_not_increased"]
        )

    def test_diversity_loss_is_bounded_in_percentage_points(self) -> None:
        terminal = metric_fixture(
            nonshortcut=300,
            composition=305,
            primary=280,
            unique_rate=0.90,
            coverage_rate=0.70,
        )
        full = metric_fixture(
            nonshortcut=311,
            composition=316,
            primary=290,
            unique_rate=0.88,
            coverage_rate=0.68,
        )
        gates, _ = scientific_gates_from_metrics(
            terminal=terminal,
            full=full,
        )
        self.assertTrue(
            gates["full_unique_formula_rate_loss_at_most_2pp"]
        )
        self.assertTrue(
            gates["full_element_coverage_rate_loss_at_most_2pp"]
        )
        full["unique_formula_rate"] = 0.879
        gates, _ = scientific_gates_from_metrics(
            terminal=terminal,
            full=full,
        )
        self.assertFalse(
            gates["full_unique_formula_rate_loss_at_most_2pp"]
        )

    def test_state_budget_is_not_a_scientific_gate(self) -> None:
        terminal = metric_fixture(
            nonshortcut=300,
            composition=305,
            primary=280,
        )
        full = metric_fixture(
            nonshortcut=311,
            composition=316,
            primary=290,
        )
        full["mechanism_telemetry"][
            "semantic_state_work_report_only"
        ] = {"max_states_created": 99_000_000}
        gates, _ = scientific_gates_from_metrics(
            terminal=terminal,
            full=full,
        )
        self.assertTrue(all(gates.values()))
        self.assertNotIn("state", " ".join(gates))

    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(0, 1), 1.0)
        self.assertLess(exact_mcnemar_p(0, 10), 0.01)


if __name__ == "__main__":
    unittest.main()

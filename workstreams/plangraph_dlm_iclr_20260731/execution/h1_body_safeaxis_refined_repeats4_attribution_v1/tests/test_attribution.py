from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "analyze.py"
SPEC = importlib.util.spec_from_file_location("r03h_attribution", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def attempt(
    *,
    ordinal: int,
    applicable: bool,
    e_hull,
    strict: bool,
    meta: bool,
):
    hull = {
        "applicable": applicable,
        "generation_rerun": False,
        "refinement_rerun": False,
        "chgnet_rerun": False,
        "direct_metrics_rerun": False,
        "novelty_rerun": False,
        "sample_retry_or_replacement_used": False,
    }
    if applicable:
        hull.update(
            {
                "chemsys": "A-B",
                "source_e_above_hull": e_hull,
                "recomputed_e_above_hull": e_hull,
            }
        )
    else:
        hull["reason"] = "not_novel_unique"
    return {
        "attempt_id": f"x-{ordinal}",
        "generation_ordinal": ordinal,
        "generation_status": "succeeded",
        "evaluation_status": "evaluated" if applicable else "not_applicable",
        "retry_or_replacement_used": False,
        "hull_recompute": hull,
        "metrics": {
            "novel_unique": applicable,
            "strict_full_sun": strict,
            "meta_full_sun": meta,
        },
    }


def classified_state(state: str, ordinal: int = 0):
    applicable = state != "ineligible"
    e_hull = {
        "ineligible": None,
        "residual_unknown": None,
        "strict": 0.0,
        "meta_only": 0.05,
        "above_meta": 0.2,
    }[state]
    return {
        "attempt_id": f"x-{ordinal}",
        "generation_ordinal": ordinal,
        "state": state,
        "applicable_novel_unique": applicable,
        "strict_full_sun": state == "strict",
        "meta_full_sun": state in {"strict", "meta_only"},
        "e_above_hull": e_hull,
        "source_e_above_hull": e_hull,
        "chemsys": "A-B" if applicable else None,
        "composition": {"A": 1.0, "B": 1.0} if applicable else None,
        "elements": ["A", "B"] if applicable else None,
        "element_count": 2 if applicable else None,
        "atom_count": 2.0 if applicable else None,
        "ineligible_reason": None if applicable else "not_novel_unique",
        "generation_status": "succeeded",
        "evaluation_status": "evaluated" if applicable else "not_applicable",
    }


class AttributionTests(unittest.TestCase):
    def test_state_boundaries_are_frozen(self) -> None:
        cases = [
            ("strict", True, 0.0, True, True),
            ("meta_only", True, 0.05, False, True),
            ("above_meta", True, 0.1000001, False, False),
            ("residual_unknown", True, None, False, False),
            ("ineligible", False, None, False, False),
        ]
        for expected, applicable, e_hull, strict, meta in cases:
            with self.subTest(expected=expected):
                row = attempt(
                    ordinal=0,
                    applicable=applicable,
                    e_hull=e_hull,
                    strict=strict,
                    meta=meta,
                )
                result = MODULE.classify_attempt(
                    row,
                    strict_threshold=0.0,
                    meta_threshold=0.1,
                    composition={"A": 1, "B": 1} if applicable else None,
                )
                self.assertEqual(result["state"], expected)

    def test_discordance_mechanisms(self) -> None:
        self.assertEqual(
            MODULE.discordance_mechanism("ineligible"),
            "novel_unique_eligibility",
        )
        self.assertEqual(
            MODULE.discordance_mechanism("residual_unknown"),
            "residual_hull_unknown",
        )
        self.assertEqual(
            MODULE.discordance_mechanism("meta_only"),
            "finite_hull_threshold_crossing",
        )
        self.assertEqual(
            MODULE.discordance_mechanism("above_meta"),
            "finite_hull_threshold_crossing",
        )

    def test_quantile_uses_linear_interpolation(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(MODULE.quantile(values, 0.0), 0.0)
        self.assertEqual(MODULE.quantile(values, 0.5), 15.0)
        self.assertEqual(MODULE.quantile(values, 1.0), 30.0)

    def test_endpoint_decomposition(self) -> None:
        transitions = [
            ("meta_only", "strict"),
            ("above_meta", "strict"),
            ("ineligible", "strict"),
            ("strict", "meta_only"),
            ("strict", "ineligible"),
            ("residual_unknown", "residual_unknown"),
        ]
        pairs = []
        for ordinal, (control_state, candidate_state) in enumerate(transitions):
            control = classified_state(control_state, ordinal)
            candidate = classified_state(candidate_state, ordinal)
            pairs.append(
                {
                    "repeat": 0,
                    "ordinal": ordinal,
                    "chemsys": "A-B",
                    "atom_count": 2.0,
                    "element_count": 2,
                    "control": control,
                    "candidate": candidate,
                }
            )
        report = MODULE.build_endpoint_report(pairs, "strict_full_sun")
        self.assertEqual(report["candidate_only"], 3)
        self.assertEqual(report["control_only"], 2)
        self.assertEqual(report["candidate_minus_control"], 1)
        self.assertEqual(
            report["net_effect_by_mechanism"],
            {
                "finite_hull_threshold_crossing": 1,
                "novel_unique_eligibility": 0,
            },
        )

    def test_build_pairs_preserves_full_mapping(self) -> None:
        arms = {}
        for repeat in MODULE.REPEATS:
            for arm in MODULE.ARMS:
                arms[f"r{repeat}_{arm}"] = [
                    classified_state("ineligible", ordinal)
                    for ordinal in range(256)
                ]
        pairs = MODULE.build_pairs(arms)
        self.assertEqual(len(pairs), 1024)
        self.assertEqual(pairs[0]["state_transition"], "ineligible->ineligible")
        self.assertEqual(pairs[-1]["repeat"], 3)
        self.assertEqual(pairs[-1]["ordinal"], 255)

    def test_recurrence_is_by_ordinal_across_repeats(self) -> None:
        events = [
            {"ordinal": 7, "repeat": 0, "chemsys": "A-B"},
            {"ordinal": 7, "repeat": 2, "chemsys": "A-B"},
            {"ordinal": 9, "repeat": 1, "chemsys": "C-D"},
        ]
        report = MODULE.recurrence_report(events)
        self.assertEqual(report["event_count"], 3)
        self.assertEqual(report["distinct_ordinals"], 2)
        self.assertEqual(
            report["ordinal_recurrence_histogram"], {"1": 1, "2": 1}
        )
        self.assertEqual(report["recurrent_ordinals"][0]["ordinal"], 7)


if __name__ == "__main__":
    unittest.main()

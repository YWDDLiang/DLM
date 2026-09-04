import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_full_mp20_transaction_sources",
    ROOT / "scripts" / "freeze_full_mp20_transaction_sources.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import full MP20 transaction source freezer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def answer(species):
    text, _ = arrays_to_dynamic_answer(
        lengths=[4.0, 4.5, 5.0],
        angles=[90.0, 90.0, 90.0],
        species=species,
        frac_coords=[
            [index / len(species), (index + 1) / (len(species) + 1), 0.25]
            for index in range(len(species))
        ],
    )
    return text


def paired_rows(source_idx, species, program, *, source="fixture"):
    counts = []
    elements = []
    for symbol in species:
        if symbol not in elements:
            elements.append(symbol)
            counts.append(0)
        counts[elements.index(symbol)] += 1
    plan = {"N": len(species), "elements": elements, "counts": counts}
    teacher = answer(species)
    prompt = f"prompt-{source_idx}"
    sft = {
        "source_row_idx": source_idx,
        "source_split": "train",
        "plan_state": dict(plan),
        "prompt": prompt,
        "answer": teacher,
        "source_answer": teacher,
    }
    plan_row = {
        "source_row_idx": source_idx,
        "plan_state": dict(plan),
        "prompt": prompt,
        "teacher_answer": teacher,
        "species_program": list(program),
        "source": source,
    }
    return sft, plan_row


class FullMP20TransactionSourceFreezerTest(unittest.TestCase):
    def fixture(self):
        pairs = [
            paired_rows(0, ["Na", "O", "Na", "Cl"], ["Cl", "Na", "O"]),
            paired_rows(1, ["Na", "O", "Na", "Cl"], ["Cl", "Na", "O"]),
            paired_rows(2, ["Na", "O", "Na", "Cl"], ["Cl", "Na", "O"]),
            paired_rows(3, ["Li", "Li"], ["Li"]),
            paired_rows(4, ["Li", "Li"], ["Li"]),
            paired_rows(5, ["Li", "Li"], ["Li"]),
        ]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def test_multispecies_stage_assignment_uses_actual_teacher_slots(self):
        sft, plans = self.fixture()
        rows = MODULE.build_full_mp20_transaction_sources(
            sft, plans, common_seed=17
        )
        self.assertEqual([row["source_row_idx"] for row in rows], list(range(6)))
        self.assertEqual([row["sample_idx"] for row in rows], list(range(6)))
        self.assertEqual(
            [row["deployment_stage"] for row in rows[:3]],
            ["cell", "anchor_second", "anchor_first"],
        )
        # Program Cl -> Na -> O maps to actual teacher slots 3 -> 0 -> 1.
        self.assertEqual(rows[0]["species_program_anchor_slots"], [3, 0, 1])
        self.assertIsNone(rows[1]["anchor_slot"])
        self.assertEqual(rows[1]["teacher_anchor_slot"], 0)
        self.assertEqual(rows[1]["anchor_symbol"], "Na")
        self.assertIsNone(rows[1]["active_positions"])
        self.assertEqual(
            rows[1]["active_positions_resolution"],
            "resolve_from_generated_body_species",
        )
        self.assertIsNone(rows[2]["anchor_slot"])
        self.assertEqual(rows[2]["teacher_anchor_slot"], 3)
        self.assertEqual(rows[2]["anchor_symbol"], "Cl")
        self.assertIsNone(rows[2]["active_positions"])
        self.assertEqual(
            [item["stage"] for item in rows[0]["remaining_reference_stages"]],
            ["anchor_second", "anchor_first"],
        )
        self.assertTrue(
            all(
                item["executor"] == "frozen_reference_dlm"
                for item in rows[0]["remaining_reference_stages"]
            )
        )
        self.assertEqual(
            rows[0]["remaining_reference_stages"][0]["common_random_seed"],
            17 + 1_009,
        )
        self.assertEqual(
            [item["stage"] for item in rows[1]["remaining_reference_stages"]],
            ["anchor_first"],
        )
        self.assertEqual(rows[2]["remaining_reference_stages"], [])
        self.assertTrue(all(row["source_weight"] == 1.0 for row in rows))
        self.assertTrue(all(row["outcomes_read"] is False for row in rows))
        self.assertTrue(all(row["retain_failure_placeholder"] for row in rows))
        self.assertTrue(all(row["source_marker"] == "fixture" for row in rows))

    def test_unary_anchor_stages_deterministically_share_unique_anchor(self):
        sft, plans = self.fixture()
        rows = MODULE.build_full_mp20_transaction_sources(sft, plans)
        for index in (4, 5):
            self.assertTrue(rows[index]["unary_anchor_fallback"])
            self.assertEqual(rows[index]["anchor_first_teacher_slot"], 0)
            self.assertEqual(rows[index]["anchor_second_teacher_slot"], 0)
            self.assertIsNone(rows[index]["anchor_slot"])
            self.assertEqual(rows[index]["teacher_anchor_slot"], 0)
            self.assertEqual(rows[index]["anchor_symbol"], "Li")

    def test_common_seed_is_fixed_and_stage_aware(self):
        sft, plans = self.fixture()
        first = MODULE.build_full_mp20_transaction_sources(
            sft, plans, common_seed=101
        )
        second = MODULE.build_full_mp20_transaction_sources(
            sft, plans, common_seed=101
        )
        self.assertEqual(
            [row["common_random_seed"] for row in first],
            [row["common_random_seed"] for row in second],
        )
        self.assertEqual(first[2]["common_random_seed"], 101 + 2 * 1_000_003 + 2 * 1_009)

    def test_missing_plan_row_is_rejected_without_filtering_denominator(self):
        sft, plans = self.fixture()
        with self.assertRaisesRegex(ValueError, "not contiguous|coverage mismatch"):
            MODULE.build_full_mp20_transaction_sources(sft, plans[:-1])

    def test_duplicate_source_row_is_rejected(self):
        sft, plans = self.fixture()
        plans[-1] = dict(plans[-1], source_row_idx=4)
        with self.assertRaisesRegex(ValueError, "duplicates source_row_idx 4"):
            MODULE.build_full_mp20_transaction_sources(sft, plans)

    def test_wrong_plan_is_rejected(self):
        sft, plans = self.fixture()
        plans[0] = dict(plans[0])
        plans[0]["plan_state"] = dict(plans[0]["plan_state"], N=5)
        with self.assertRaisesRegex(ValueError, "Plan mismatch"):
            MODULE.build_full_mp20_transaction_sources(sft, plans)

    def test_formal_mode_requires_full_27136_rows_but_fixture_mode_does_not(self):
        sft, plans = self.fixture()
        self.assertEqual(
            len(MODULE.build_full_mp20_transaction_sources(sft, plans)), 6
        )
        with self.assertRaisesRegex(ValueError, "formal mode requires exactly 27136"):
            MODULE.build_full_mp20_transaction_sources(sft, plans, formal=True)


if __name__ == "__main__":
    unittest.main()

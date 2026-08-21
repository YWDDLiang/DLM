import unittest

from h1a2_repro.story_panel import build_panels, stable_seed


def row(source: str, index: int, lattice: str, sg: str, volume: str):
    return {
        "source": source,
        "plan_id": f"{source}:{index}",
        "source_row": {},
        "plan_state": {
            "N": 3,
            "elements": ["Li", "O"],
            "counts": [2, 1],
            "formula": "Li2O",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": lattice,
            "spacegroup_bucket": sg,
            "volume_per_atom_bin": volume,
        },
    }


class StoryPanelTests(unittest.TestCase):
    def test_stable_seed(self) -> None:
        self.assertEqual(stable_seed(17, "a", "b"), stable_seed(17, "a", "b"))
        self.assertNotEqual(stable_seed(17, "a", "b"), stable_seed(17, "a", "c"))

    def test_panel_counts_and_paired_seeds(self) -> None:
        learned = [
            row("learned", 0, "cubic", "sg_195_230", "volpa_008_012"),
            row("learned", 1, "tetragonal", "sg_075_142", "volpa_012_016"),
        ]
        gold = [
            row("gold", 0, "orthorhombic", "sg_016_074", "volpa_016_020"),
            row("gold", 1, "hexagonal", "sg_168_194", "volpa_020_024"),
        ]
        tasks, e2_task_ids, report = build_panels(
            learned,
            gold,
            num_pairs=2,
            e2_pairs=1,
            seed=20260822,
        )
        self.assertEqual(len(tasks), 64)
        self.assertEqual(len(e2_task_ids), 24)
        self.assertEqual(report["e1_requested_tasks"], 64)

        by_key = {(task["plan_id"], task["arm"], task["replicate"]): task for task in tasks}
        for plan_id in {task["plan_id"] for task in tasks}:
            seeds = {by_key[(plan_id, arm, 0)]["scientific_seed"] for arm in ("full", "formula", "shuffle")}
            self.assertEqual(len(seeds), 1)
            original = by_key[(plan_id, "full", 0)]["plan_state"]
            shuffled = by_key[(plan_id, "shuffle", 0)]["plan_state"]
            self.assertIn("shuffle_donor_plan_id", by_key[(plan_id, "shuffle", 0)])
            self.assertTrue(
                any(
                    original[field] != shuffled[field]
                    for field in ("lattice_system", "spacegroup_bucket", "volume_per_atom_bin")
                )
            )

    def test_shuffle_requires_a_changed_coarse_tuple(self) -> None:
        learned = [row("learned", 0, "cubic", "sg_195_230", "volpa_008_012")]
        gold = [row("gold", 0, "cubic", "sg_195_230", "volpa_008_012")]
        with self.assertRaisesRegex(ValueError, "no donor with a different"):
            build_panels(learned, gold, num_pairs=1, e2_pairs=1)

    def test_requires_requested_number_of_matches(self) -> None:
        learned = [row("learned", 0, "cubic", "sg_195_230", "volpa_008_012")]
        gold = [row("gold", 0, "cubic", "sg_195_230", "volpa_008_012")]
        with self.assertRaises(ValueError):
            build_panels(learned, gold, num_pairs=2, e2_pairs=1)


if __name__ == "__main__":
    unittest.main()

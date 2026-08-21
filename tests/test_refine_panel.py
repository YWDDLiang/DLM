import unittest

from h1a2_repro.refine_panel import select_refinement_panel


class RefinePanelTests(unittest.TestCase):
    def test_failures_remain_in_requested_denominator(self) -> None:
        tasks = [
            {
                "task_id": "p/learned/full/r0",
                "pair_id": "p",
                "plan_id": "learned:0",
                "plan_source": "learned",
                "arm": "full",
                "replicate": 0,
                "scientific_seed": 11,
            },
            {
                "task_id": "p/learned/formula/r0",
                "pair_id": "p",
                "plan_id": "learned:0",
                "plan_source": "learned",
                "arm": "formula",
                "replicate": 0,
                "scientific_seed": 11,
            },
        ]
        graphs = [{"graph": 1}]
        accepted = [{"task_id": "p/learned/full/r0", "pair_id": "p", "plan_source": "learned", "replicate": 0}]
        selected, metadata, ledger, report = select_refinement_panel(
            graphs,
            accepted,
            tasks,
            [task["task_id"] for task in tasks],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(report["requested_attempts"], 2)
        self.assertEqual(report["body_failures"], 1)
        self.assertFalse(ledger[1]["refine_eligible"])

    def test_graph_metadata_length_must_match(self) -> None:
        with self.assertRaises(ValueError):
            select_refinement_panel([{}], [], [], [])


if __name__ == "__main__":
    unittest.main()

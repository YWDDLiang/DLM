from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_c3fd_native_alignment_groups.py"
SPEC = importlib.util.spec_from_file_location("build_native_alignment_groups", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BuildC3FDNativeAlignmentGroupsTest(unittest.TestCase):
    def test_invalid_candidates_are_worse_than_every_valid_candidate(self):
        plans = [
            {
                "sample_idx": idx,
                "prompt": "same",
                "plan_state": {"N": 2},
            }
            for idx in range(4)
        ]
        group = {
            "group_id": "g",
            "group_ordinal": 0,
            "composition_ordinal": 0,
            "reduced_composition_identity": "Li:1|O:1",
            "prediction_checkpoint": "seed17",
            "K": 4,
            "sample_indices": [0, 1, 2, 3],
        }
        attempts = {
            idx: {"sample_idx": idx, "text": f"answer{idx}", "parsed": True}
            for idx in range(4)
        }
        raw_direct = {
            idx: {"valid": idx in {0, 1}, "comp_valid": True, "struct_valid": idx in {0, 1}}
            for idx in range(4)
        }
        raw_labels = {idx: {"ordinal": idx, "chgnet_energy_per_atom": -3.0 + idx} for idx in range(4)}
        refined_labels = {idx: {"ordinal": idx, "chgnet_energy_per_atom": -5.0 + idx, "reconstructed": True} for idx in range(4)}
        rows, audit = MODULE.build_policy_groups(
            policy=82017,
            plans=plans,
            groups=[group],
            raw_attempts=attempts,
            evaluation={
                "raw_direct": raw_direct,
                "raw_labels": raw_labels,
                "refined_labels": refined_labels,
                "refined_direct": raw_direct,
            },
        )
        built = rows[0]
        self.assertTrue(built["trainable"])
        self.assertEqual(built["best_valid_candidate_index"], 0)
        targets = [candidate["target_energy_per_atom"] for candidate in built["candidates"]]
        self.assertLess(max(targets[:2]), min(targets[2:]))
        self.assertEqual(audit["groups_with_validity_variation"], 1)

    def test_no_valid_group_is_preserved_but_not_trainable(self):
        plans = [{"sample_idx": idx, "prompt": "same", "plan_state": {"N": 2}} for idx in range(4)]
        group = {"group_id": "g", "group_ordinal": 0, "composition_ordinal": 0, "reduced_composition_identity": "Li:1|O:1", "prediction_checkpoint": "seed17", "K": 4, "sample_indices": [0, 1, 2, 3]}
        attempts = {idx: {"sample_idx": idx, "text": f"a{idx}", "parsed": False} for idx in range(4)}
        direct = {idx: {"valid": False, "comp_valid": False, "struct_valid": False} for idx in range(4)}
        labels = {idx: {"ordinal": idx, "chgnet_energy_per_atom": None, "reconstructed": False} for idx in range(4)}
        rows, audit = MODULE.build_policy_groups(policy=82018, plans=plans, groups=[group], raw_attempts=attempts, evaluation={"raw_direct": direct, "raw_labels": labels, "refined_labels": labels, "refined_direct": direct})
        self.assertFalse(rows[0]["trainable"])
        self.assertIsNone(rows[0]["best_valid_candidate_index"])
        self.assertEqual(audit["groups_without_valid_anchor"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.r5_plan_state import plan_state_from_arrays
from crystal_dlm.r5_repair import choose_masked_block, labels_from_arrays, make_repair_record
from crystal_dlm.r5_verifier import VerifierWeights, extract_verifier_features, utility_score


class R5RepairVerifierTests(unittest.TestCase):
    def make_arrays(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3.0, 3.0, 3.0],
            angles=[90, 90, 90],
            species=["Li", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        return parse_dynamic_answer(answer, strict=True), answer

    def test_repair_record_chooses_geometry_block(self):
        arrays, answer = self.make_arrays()
        plan = plan_state_from_arrays(arrays)
        labels = labels_from_arrays(arrays)
        self.assertIn("a_eq_b_eq_c", labels)
        self.assertEqual(choose_masked_block(labels), "lattice+coords")
        record = make_repair_record(
            plan_state=plan,
            visible_proposal=answer,
            target="<LA_040><LB_041><LC_042>",
            violation_labels=labels,
        )
        self.assertEqual(record["masked_block"], "lattice+coords")
        self.assertIn("violation_labels:", record["prompt"])

    def test_verifier_penalizes_high_symmetry_and_rewards_metrics(self):
        arrays, _ = self.make_arrays()
        plan = plan_state_from_arrays(arrays)
        features = extract_verifier_features(
            arrays,
            plan_state=plan,
            metric_record={"energy_above_hull_per_atom": -0.01, "novel": True, "unique": True},
        )
        self.assertTrue(features["plan_match"])
        self.assertTrue(features["meta_stable"])
        score = utility_score(features, weights=VerifierWeights())
        worse = dict(features)
        worse["single_element"] = True
        worse["pbc_duplicate_count"] = 2
        self.assertGreater(score, utility_score(worse, weights=VerifierWeights()))


if __name__ == "__main__":
    unittest.main()

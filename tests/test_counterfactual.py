import unittest

from h1a2_repro.counterfactual import (
    ANCHORED_FIELDS,
    build_counterfactual_plan,
    calibrated_grounding_weight,
    choose_donors,
    geometry_relative_positions,
)


def plan(formula, lattice, sg, volume):
    return {
        "formula": formula,
        "N": 3,
        "elements": ["Li", "O"],
        "counts": [2, 1],
        "anion_framework": "oxide",
        "charge_bucket": "neutral_plausible",
        "lattice_system": lattice,
        "spacegroup_bucket": sg,
        "volume_per_atom_bin": volume,
    }


class CounterfactualTest(unittest.TestCase):
    def test_counterfactual_changes_only_structural_tuple(self):
        factual = plan("Li2O", "cubic", "sg_195_230", "volpa_010_014")
        donor = plan("Na2O", "orthorhombic", "sg_016_074", "volpa_020_024")
        changed = build_counterfactual_plan(factual, donor)
        for field in ANCHORED_FIELDS:
            self.assertEqual(changed[field], factual[field])
        self.assertEqual(changed["lattice_system"], donor["lattice_system"])
        self.assertEqual(changed["spacegroup_bucket"], donor["spacegroup_bucket"])
        self.assertEqual(changed["volume_per_atom_bin"], donor["volume_per_atom_bin"])

    def test_donor_and_geometry_count(self):
        plans = [
            plan("Li2O", "cubic", "sg_195_230", "volpa_010_014"),
            plan("Na2O", "orthorhombic", "sg_016_074", "volpa_020_024"),
        ]
        donors = choose_donors(plans, seed=17)
        self.assertEqual(donors, [1, 0])
        self.assertEqual(len(geometry_relative_positions(3)), 15)
        self.assertEqual(geometry_relative_positions(1), (1, 2, 3, 4, 5, 6, 8, 9, 10))

    def test_calibration(self):
        self.assertAlmostEqual(
            calibrated_grounding_weight(
                ce_gradient_norm=2.0,
                grounding_gradient_norm=4.0,
                target_ratio=0.1,
            ),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()


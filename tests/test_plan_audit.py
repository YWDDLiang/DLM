import unittest

from h1a2_repro.plan_audit import audit_plans, entropy, normalize_plan


def plan(formula, lattice, *, body_success=None, strict_sun=None):
    row = {
        "plan_state": {
            "formula": formula,
            "anion_framework": "oxide" if "O" in formula else "none",
            "charge_bucket": "neutral_plausible",
            "lattice_system": lattice,
            "spacegroup_bucket": "sg-a",
            "volume_per_atom_bin": "v-a",
        }
    }
    if body_success is not None or strict_sun is not None:
        row["outcomes"] = {"body_success": body_success, "strict_sun": strict_sun}
    return row


class PlanAuditTests(unittest.TestCase):
    def test_normalization_and_entropy(self) -> None:
        normalized = normalize_plan(plan("Li2O", "cubic"))
        self.assertEqual(normalized["composition_key"], "Li:2|O:1")
        self.assertEqual(normalized["N"], "3")
        self.assertAlmostEqual(entropy(["a", "a"]), 0.0)

    def test_collision_and_distance(self) -> None:
        train = [plan("Li2O", "cubic"), plan("NaCl", "cubic")]
        generated = [
            plan("Li2O", "cubic", body_success=True, strict_sun=True),
            plan("Li2O", "tetragonal", body_success=False, strict_sun=False),
        ]
        report = audit_plans(generated, train)
        self.assertEqual(report["collisions"]["formula_exact"], 2)
        self.assertEqual(report["collisions"]["full_tuple_exact"], 1)
        self.assertEqual(report["nearest_train_tuple_hamming"]["counts"], {0: 1, 1: 1})
        oxide = report["downstream_conversion"]["family"]["oxide"]
        self.assertEqual(oxide["body_success"]["numerator"], 1)


if __name__ == "__main__":
    unittest.main()

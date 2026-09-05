from types import SimpleNamespace
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "label_programmed_paths", Path(__file__).resolve().parents[1] / "scripts" / "label_programmed_paths.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
label_record, EV_A3_TO_GPA = MODULE.label_record, MODULE.EV_A3_TO_GPA


class Structure:
    num_sites = 2
    composition = "H2"
    lattice = SimpleNamespace(matrix=np.eye(3) * 3)
    frac_coords = np.array([[0., 0., 0.], [.5, .5, .5]])
    def as_dict(self):
        return {"sites": 2}


class Model:
    def predict_structure(self, structure, task):
        return {"e": 3., "f": np.ones((2, 3)), "s": np.eye(3)}


class Optimizer:
    def __init__(self, stress_GPa=.2, converged=True):
        self.stress, self.converged = stress_GPa, converged
    def relax(self, structure, **kwargs):
        assert kwargs["relax_cell"] and kwargs["ase_filter"] == "FrechetCellFilter"
        return {"final_structure": Structure(),
                "trajectory": SimpleNamespace(energies=[6., 4.], forces=[np.zeros((2, 3))],
                                               stresses=[np.eye(3) * self.stress / EV_A3_TO_GPA]),
                "optimizer_status": {"steps": 7, "converged": self.converged}}


class ProgrammedPathLabelTest(unittest.TestCase):
    def run_case(self, optimizer):
        record = {"trajectory_id": "x:0", "group_id": "x", "source_split": "train", "success": True}
        return label_record(record, model=Model(), optimizer=optimizer, structure_factory=lambda _: Structure())

    def test_same_units_and_two_energy_terms(self):
        result = self.run_case(Optimizer())
        self.assertTrue(result["verified"])
        self.assertEqual(result["raw_energy"], 3.)
        self.assertEqual(result["terminal_energy"], 2.)
        self.assertEqual(result["gap"], 1.)
        self.assertAlmostEqual(result["terminal"]["stress_max_GPa"], .2)

    def test_finite_but_not_converged_is_not_verified(self):
        result = self.run_case(Optimizer(stress_GPa=2.))
        self.assertEqual(result["status"], "not_converged")
        self.assertFalse(result["verified"])
        self.assertIsNotNone(result["terminal_energy"])

    def test_optimizer_stop_is_not_fabricated(self):
        result = self.run_case(Optimizer(converged=None))
        self.assertFalse(result["verified"])
        self.assertIsNone(result["optimizer_converged"])

    def test_generation_failure_has_no_imputed_energy(self):
        result = label_record({"success": False}, model=None, optimizer=None)
        self.assertEqual(result["status"], "generation_failure")
        self.assertIsNone(result["raw_energy"])

    def test_colliding_terminal_cannot_be_verified(self):
        class BadOptimizer(Optimizer):
            def relax(self, *args, **kwargs):
                result = super().relax(*args, **kwargs)
                result["final_structure"].frac_coords = np.zeros((2, 3))
                return result
        result = self.run_case(BadOptimizer())
        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "invalid_terminal")
        self.assertIsNotNone(result["terminal_energy"])
        self.assertEqual(result["actual_steps"], 7)

    def test_negative_gap_is_not_clipped(self):
        class UphillOptimizer(Optimizer):
            def relax(self, *args, **kwargs):
                result = super().relax(*args, **kwargs)
                result["trajectory"].energies[-1] = 8.
                return result
        result = self.run_case(UphillOptimizer())
        self.assertEqual(result["gap"], -1.)
        self.assertEqual(result["status"], "relaxation_energy_increased")
        self.assertFalse(result["verified"])

    def test_raw_and_trajectory_energy_protocol_mismatch(self):
        class MismatchOptimizer(Optimizer):
            def relax(self, *args, **kwargs):
                result = super().relax(*args, **kwargs)
                result["trajectory"].energies[0] = 60.
                return result
        result = self.run_case(MismatchOptimizer())
        self.assertEqual(result["status"], "energy_protocol_mismatch")
        self.assertFalse(result["verified"])


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "label_programmed_paths", Path(__file__).resolve().parents[1] / "scripts" / "label_programmed_paths.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
label_record, EV_A3_TO_GPA = MODULE.label_record, MODULE.EV_A3_TO_GPA


class PurposeContractTests(unittest.TestCase):
    def test_expert_edit_has_explicit_training_or_development_provenance(self):
        for split in ('train', 'dev'):
            MODULE.validate_record_purpose({'source_split': split, 'purpose': 'expert_edit',
                                            'endpoint': 'expert_quantized'}, 'expert_edit')
        with self.assertRaises(ValueError):
            MODULE.validate_record_purpose({'source_split': 'evaluation', 'purpose': 'expert_edit',
                                            'endpoint': 'expert_quantized'}, 'expert_edit')
        with self.assertRaises(ValueError):
            MODULE.validate_record_purpose({'source_split': 'train', 'endpoint': 'expert_quantized'}, 'expert_edit')
        with self.assertRaises(ValueError):
            MODULE.validate_record_purpose({'source_split': 'train', 'endpoint': 'expert_quantized'}, 'train')


class PeriodicGeometryProtocolTests(unittest.TestCase):
    def test_skew_cell_contact_outside_legacy_image_shell_is_rejected(self):
        from pymatgen.core import Lattice, Structure
        lattice = Lattice.from_parameters(1.,3.1,5.,90.,90.,2.)
        crystal = Structure(lattice,['Na'],[[0.,0.,0.]])
        shifts = np.stack(np.meshgrid(*([np.arange(-2,3)]*3), indexing='ij'),axis=-1).reshape(-1,3)
        self.assertGreaterEqual(float(np.linalg.norm(shifts[np.any(shifts!=0,axis=1)]@lattice.matrix,axis=-1).min()),.5)
        self.assertLess(float(np.linalg.norm(np.array([-3,1,0])@lattice.matrix)),.5)
        with self.assertRaisesRegex(ValueError, '0.5 Angstrom'):
            MODULE.validate_structure_geometry(crystal)

    def test_basis_and_periodic_translation_preserve_certified_support(self):
        from pymatgen.core import Lattice, Structure
        lattice = np.eye(3)*4
        unimodular = np.array([[1,0,0],[10,1,0],[3,2,1]])
        coordinates = np.array([[0.,0.,0.],[.5,.5,.5]])
        changed = unimodular@lattice
        transformed = coordinates@np.linalg.inv(unimodular)+np.array([3.,-2.,4.])
        for matrix,coords in ((lattice,coordinates),(changed,transformed)):
            self.assertGreaterEqual(MODULE.validate_structure_geometry(Structure(Lattice(matrix),['Na','Cl'],coords)),.5)

    def test_contact_work_budget_is_engineering_unknown_not_invalid_geometry(self):
        from pymatgen.core import Lattice, Structure
        crystal = Structure(Lattice.cubic(4),['Na'],[[0.,0.,0.]])
        with patch.dict(MODULE.LABEL_GEOMETRY_PROTOCOL, {'max_pair_images':0}):
            result = MODULE.label_record({'success':True}, model=None, optimizer=None, structure_factory=lambda _:crystal)
        self.assertEqual(result['status'],'worker_error')
        self.assertIsNone(result['terminal_energy'])

    def test_lll_runtime_or_memory_failure_is_unknown_at_raw_and_terminal_stages(self):
        from pymatgen.core import Lattice
        for terminal in (False, True):
            for error_type in (RuntimeError, MemoryError):
                with self.subTest(terminal=terminal, error_type=error_type.__name__):
                    failures = ([Lattice.cubic(3), error_type('reduction could not finish')] if terminal else
                                error_type('reduction could not finish'))
                    with patch.object(Lattice, 'get_lll_reduced_lattice', side_effect=failures):
                        result = MODULE.label_record({'success': True}, model=Model(), optimizer=Optimizer(),
                                                     structure_factory=lambda _: Structure())
                    self.assertEqual(result['status'], 'worker_error')
                    self.assertFalse(result['verified'])

    def test_near_integral_lll_error_cannot_change_original_threshold_classification(self):
        from pymatgen.core import Lattice, Structure
        cases = ((.50000001, .49999998, False), (.49999997, .5, True))
        for original_length, returned_length, truly_short in cases:
            with self.subTest(original_length=original_length, returned_length=returned_length):
                original = Structure(Lattice(np.diag([original_length, 4., 4.])), ['Na'], [[0., 0., 0.]])
                returned = Lattice(np.diag([returned_length, 4., 4.]))
                before = original.as_dict()
                with patch.object(Lattice, 'get_lll_reduced_lattice', return_value=returned):
                    try:
                        minimum = MODULE.validate_structure_geometry(original)
                    except MODULE.GeometryCertificationUnavailable:
                        pass  # Numerical uncertainty is a valid engineering outcome.
                    except ValueError:
                        self.assertTrue(truly_short, 'approximate LLL output fabricated a short contact')
                    else:
                        self.assertFalse(truly_short, 'approximate LLL output hid an original short contact')
                        self.assertGreaterEqual(minimum, .5 - 1e-8)
                self.assertEqual(original.as_dict(), before)


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
        return label_record(record, model=Model(), optimizer=optimizer, structure_factory=lambda _: Structure(),
                            terminal_energy_checker=lambda *args: {"status": "consistent"})

    def test_terminal_consistency_failure_retains_energy_but_withholds_verification(self):
        result = label_record({"success": True}, model=Model(), optimizer=Optimizer(),
                              structure_factory=lambda _: Structure(),
                              terminal_energy_checker=lambda *args: {"status": "inconsistent"})
        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "terminal_consistency_unverified")
        self.assertEqual(result["terminal_energy"], 2.)

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

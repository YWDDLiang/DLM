import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_potential_closure_phase0",
    ROOT / "scripts" / "audit_potential_closure_phase0.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import Phase 0 audit")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase0AuditTest(unittest.TestCase):
    def test_stable_retention_uses_paired_proxy(self):
        rows = [
            {
                "cached_dft_e_hull_eV_per_atom": 0.00,
                "quantized_proxy_e_hull_eV_per_atom": 0.01,
            },
            {
                "cached_dft_e_hull_eV_per_atom": 0.08,
                "quantized_proxy_e_hull_eV_per_atom": 0.11,
            },
            {
                "cached_dft_e_hull_eV_per_atom": 0.20,
                "quantized_proxy_e_hull_eV_per_atom": 0.05,
            },
        ]
        result = MODULE.stable_retention(rows, threshold=0.1)
        self.assertEqual(result["continuous_stable"], 2)
        self.assertEqual(result["retained"], 1)
        self.assertEqual(result["retention"], 0.5)

    def test_stratified_indices_are_deterministic_and_unique(self):
        rows = [
            {"status": "ok", "num_atoms": 2 + index % 17, "arity": 1 + index % 4}
            for index in range(80)
        ]
        first = MODULE.stratified_indices(rows, count=32, seed=17)
        second = MODULE.stratified_indices(rows, count=32, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(len(set(first)), 32)

    def test_describe_ignores_nonfinite_values(self):
        result = MODULE.describe([1.0, float("nan"), 3.0])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["median"], 2.0)

    def test_report_uses_subset_for_potential_coverage(self):
        rows = []
        for index in range(4):
            rows.append(
                {
                    "status": "ok",
                    "num_atoms": 2,
                    "arity": 1,
                    "answer_token_count": 15,
                    "expected_token_count": 15,
                    "exact_composition": True,
                    "exact_species_order": True,
                    "finite_geometry": True,
                    "continuous_frozen_comp_valid": True,
                    "quantized_frozen_comp_valid": True,
                    "continuous_frozen_struct_valid": True,
                    "quantized_frozen_struct_valid": True,
                    "continuous_frozen_direct_joint": True,
                    "quantized_frozen_direct_joint": True,
                    "chgnet_pair_known": index < 2,
                    "quantized_minus_continuous_energy_eV_per_atom": 0.001,
                    "continuous_chgnet": {
                        "force_rms_eV_per_A": 1.0,
                        "force_max_eV_per_A": 2.0,
                        "stress_frobenius_GPa": 3.0,
                    },
                    "quantized_chgnet": {
                        "force_rms_eV_per_A": 1.1,
                        "force_max_eV_per_A": 2.1,
                        "stress_frobenius_GPa": 3.1,
                    },
                    "force_direction_cosine": 0.99,
                    "stress_direction_cosine": 0.98,
                    "hydrostatic_stress_delta_GPa": 0.01,
                    "continuous_volume_A3": 10.0,
                    "quantized_volume_A3": 10.1,
                    "continuous_minimum_distance_A": 1.0,
                    "quantized_minimum_distance_A": 1.01,
                    "metric_relative_frobenius": 0.01,
                    "angle_max_abs_delta_degree": 0.2,
                    "length_max_abs_delta_A": 0.04,
                    "cached_dft_e_hull_eV_per_atom": 0.05,
                    "quantized_proxy_e_hull_eV_per_atom": 0.051,
                }
            )
        report = MODULE.build_report(
            rows,
            fast_indices=[0, 1],
            selection_seed=17,
            expected_rows=4,
            median_energy_limit=0.015,
            validity_drop_limit=0.01,
            retention_limit=0.60,
        )
        self.assertEqual(report["chgnet_pair_coverage"], 1.0)
        self.assertTrue(report["formal_training_authorized_by_phase0"])


if __name__ == "__main__":
    unittest.main()

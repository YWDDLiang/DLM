import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_potential_closure_native_raw",
    ROOT / "scripts" / "evaluate_potential_closure_native_raw.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import native raw evaluator")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(ordinal, energy, cluster="A:1|O:2", valid=True):
    return {
        "ordinal": ordinal,
        "energy_known": energy is not None,
        "energy_eV_per_atom": energy,
        "force_known": energy is not None,
        "force_rms_eV_per_A": None if energy is None else abs(energy) + 1.0,
        "forces_eV_per_A": None if energy is None else [[energy + 1.0, 0.0, 0.0]],
        "stress_known": energy is not None,
        "stress_frobenius_GPa": None if energy is None else abs(energy) + 2.0,
        "stress_GPa": None if energy is None else [[energy + 2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "hydrostatic_stress_GPa": None if energy is None else (energy + 2.0) / 3.0,
        "composition_cluster": cluster,
        "species_order": ["A"],
        "valid": valid,
    }


class PotentialClosureNativeRawTest(unittest.TestCase):
    def test_generation_contract_retains_failed_attempts(self):
        rows = [
            {"ordinal": index, "attempt_id": f"a-{index}", "status": "failed"}
            for index in range(4)
        ]
        MODULE.validate_generation_rows(rows, denominator=4)
        with self.assertRaisesRegex(ValueError, "retain all"):
            MODULE.validate_generation_rows(rows[:-1], denominator=4)

    def test_cluster_bootstrap_is_deterministic_and_composition_weighted(self):
        values = [("duplicated", -4.0), ("duplicated", 2.0), ("single", -3.0)]
        first = MODULE.cluster_bootstrap_mean_ci(values, seed=17, replicates=1000)
        second = MODULE.cluster_bootstrap_mean_ci(values, seed=17, replicates=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)
        self.assertAlmostEqual(first["point_mean"], -2.0)

    def test_paired_effect_keeps_unknown_in_fixed_denominator(self):
        control = [record(0, 1.0), record(1, 2.0), record(2, None), record(3, 4.0)]
        potential = [record(0, 0.8), record(1, 1.7), record(2, 2.5), record(3, 3.9)]
        result = MODULE.paired_effect(
            potential,
            control,
            comparator="closure_control",
            denominator=4,
            bootstrap_seed=11,
            bootstrap_replicates=200,
        )
        self.assertEqual(result["energy"]["paired_known"], 3)
        self.assertEqual(result["energy"]["unknown_from_fixed_denominator"], 1)
        self.assertAlmostEqual(result["energy"]["delta_eV_per_atom"]["median"], -0.2)
        self.assertEqual(result["energy"]["lower_fraction"], 1.0)

    def test_continuation_gate_uses_only_registered_raw_conditions(self):
        energy = {
            "energy": {
                "delta_eV_per_atom": {"median": -0.02},
                "composition_cluster_bootstrap_mean_95ci": {"ci95_upper": -0.001},
            }
        }
        result = MODULE.continuation_gate(
            potential_direct={"requested": 256, "composition_valid": 250, "direct_joint": 252},
            control_direct={"requested": 256, "composition_valid": 256, "direct_joint": 254},
            primary_energy=energy,
        )
        self.assertTrue(result["continue_to_conditional_full_evaluator"])
        result = MODULE.continuation_gate(
            potential_direct={"requested": 256, "composition_valid": 250, "direct_joint": 251},
            control_direct={"requested": 256, "composition_valid": 256, "direct_joint": 254},
            primary_energy=energy,
        )
        self.assertFalse(result["gates"]["direct_drop_le_0p01"])

    def test_fast_direct_loader_preserves_all_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation.jsonl"
            generation.write_text(
                "".join(
                    json.dumps(
                        {
                            "ordinal": index,
                            "attempt_id": f"attempt-{index}",
                            "status": "failed" if index == 1 else "succeeded",
                        }
                    )
                    + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            direct = root / "direct"
            direct.mkdir()
            (direct / "attempt_metrics.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "attempt_id": f"attempt-{index}",
                            "comp_valid": index != 1,
                            "struct_valid": index != 1,
                            "valid": index != 1,
                        }
                    )
                    + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            (direct / "report.json").write_text(
                json.dumps(
                    {
                        "attempts": 3,
                        "generation_succeeded": 2,
                        "comp_valid_count": 2,
                        "struct_valid_count": 2,
                        "valid_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            counts, valid = MODULE.load_direct(generation, direct, denominator=3)
            self.assertEqual(counts["requested"], 3)
            self.assertEqual(valid, {0: True, 1: False, 2: True})

    def test_slurm_contract_is_single_gpu_and_fast_raw_only(self):
        text = (ROOT / "slurm" / "188_potential_closure_native_stream17_raw_eval.sbatch").read_text()
        self.assertIn("--cpus-per-task=4", text)
        self.assertIn("gpu:NVIDIAA800-SXM4-80GB:1", text)
        self.assertIn("--mem=160G", text)
        self.assertIn("--device cuda:0 --batch-size 16", text)
        self.assertEqual(text.count("run_crysllmgen_validity_fast.py"), 1)
        self.assertIn("run_direct BS", text)
        self.assertIn("run_direct closure_control", text)
        self.assertIn("run_direct potential_closed", text)
        self.assertNotIn("run_full_reconstructed_eval.py", text)
        self.assertNotIn("refine_dlm_with_crysllmgen.py", text)


if __name__ == "__main__":
    unittest.main()

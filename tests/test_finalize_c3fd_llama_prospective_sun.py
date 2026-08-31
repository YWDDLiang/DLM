import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_c3fd_llama_prospective_sun",
    ROOT / "scripts/finalize_c3fd_llama_prospective_sun.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import prospective SUN finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProspectiveSUNFinalizerTest(unittest.TestCase):
    @staticmethod
    def evaluated_rows(prefix, *, energy_offset=0.0):
        return [
            {
                "ordinal": index,
                "attempt_id": f"{prefix}-{index}",
                "reconstructed": True,
                "chemsys": "H-O",
                "chgnet_relaxation_known": True,
                "official_hull_status": "known",
                "chgnet_energy_per_atom": float(index) + energy_offset,
            }
            for index in range(MODULE.ATTEMPTS)
        ]

    @staticmethod
    def generation_rows(prefix):
        return [
            {
                "ordinal": index,
                "sample_idx": index,
                "attempt_id": f"{prefix}-{index}",
                "plan_state": {
                    "N": 3,
                    "elements": ["H", "O"],
                    "counts": [2, 1],
                },
            }
            for index in range(MODULE.ATTEMPTS)
        ]

    def test_stream_average_uses_common_compositions(self):
        result = MODULE.average_streams([{0: 1.0, 1: 3.0}, {0: 3.0, 2: 9.0}])
        self.assertEqual(result, {0: 2.0})

    def test_asymmetric_missing_row_does_not_shift_later_pair(self):
        f_rows = self.evaluated_rows("f")
        m_rows = self.evaluated_rows("m", energy_offset=1.0)
        f_generation = self.generation_rows("f")
        m_generation = self.generation_rows("m")
        f_rows[14].update(
            reconstructed=False,
            chemsys=None,
            chgnet_relaxation_known=False,
            official_hull_status="not_reconstructed",
            chgnet_energy_per_atom=None,
        )
        f_generation[14]["plan_state"] = None
        f_generation[15]["plan_state"] = {
            "N": 2,
            "elements": ["Li", "O"],
            "counts": [1, 1],
        }
        f_rows[15]["chemsys"] = "Li-O"
        m_generation[15]["plan_state"] = {
            "N": 2,
            "elements": ["Li", "O"],
            "counts": [1, 1],
        }
        m_rows[15]["chemsys"] = "Li-O"
        f_rows = MODULE.attach_requested_identity(
            f_rows, f_generation, label="F"
        )
        m_rows = MODULE.attach_requested_identity(
            m_rows, m_generation, label="M"
        )

        result = MODULE.paired_stream_delta(
            f_rows,
            m_rows,
            field="chgnet_energy_per_atom",
            require_hull_known=False,
        )

        self.assertNotIn(14, result)
        self.assertEqual(result[15], 1.0)
        self.assertEqual(len(result), MODULE.ATTEMPTS - 1)

    def test_genuine_requested_exact_composition_mismatch_fails(self):
        f_rows = MODULE.attach_requested_identity(
            self.evaluated_rows("f"), self.generation_rows("f"), label="F"
        )
        m_generation = self.generation_rows("m")
        m_generation[14]["plan_state"] = {
            "N": 6,
            "elements": ["H", "O"],
            "counts": [4, 2],
        }
        m_rows = MODULE.attach_requested_identity(
            self.evaluated_rows("m", energy_offset=1.0),
            m_generation,
            label="M",
        )

        with self.assertRaisesRegex(ValueError, "exact composition changed at 14"):
            MODULE.paired_stream_delta(
                f_rows,
                m_rows,
                field="chgnet_energy_per_atom",
                require_hull_known=False,
            )

    def test_sun_rates_keep_fixed_requested_denominator(self):
        report = {
            "counts": {
                "raw_attempts": MODULE.ATTEMPTS,
                "reconstructed": MODULE.ATTEMPTS - 1,
                "novel": 3,
                "unique_representatives": 2,
                "novel_unique": 2,
                "hull_known_reconstructed": MODULE.ATTEMPTS - 2,
                "hull_unknown_reconstructed": 1,
                "strict_stable_all_hull_known": 2,
                "strict_sun": 1,
                "meta_stable_all_hull_known": 3,
                "meta_sun": 2,
            },
            "direct": {"joint_valid": MODULE.ATTEMPTS - 1},
        }

        summary = MODULE.summarize_cell("prospective", "refined", 17, "F", report)

        self.assertEqual(summary["requested"], MODULE.ATTEMPTS)
        self.assertEqual(summary["strict_sun"], 1)
        self.assertEqual(summary["strict_sun_rate"], 1 / MODULE.ATTEMPTS)
        self.assertEqual(summary["meta_sun_rate"], 2 / MODULE.ATTEMPTS)

        report["counts"]["raw_attempts"] = MODULE.ATTEMPTS - 1
        with self.assertRaisesRegex(ValueError, "denominator changed"):
            MODULE.summarize_cell("prospective", "refined", 17, "F", report)

    def test_bootstrap_is_deterministic(self):
        first = MODULE.bootstrap({0: -1.0, 1: 1.0}, "x")
        second = MODULE.bootstrap({0: -1.0, 1: 1.0}, "x")
        self.assertEqual(first, second)

    def test_dlm_raw_endpoint_is_explicit(self):
        source = (
            ROOT / "scripts/finalize_c3fd_llama_prospective_sun.py"
        ).read_text(encoding="utf-8")
        self.assertIn('(\"raw\", \"chgnet_energy_per_atom\", False)', source)
        self.assertIn("selection_retry_replacement_rerank", source)


if __name__ == "__main__":
    unittest.main()

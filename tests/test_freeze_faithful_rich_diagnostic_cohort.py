import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_faithful_rich_diagnostic_cohort",
    ROOT / "scripts" / "freeze_faithful_rich_diagnostic_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan(index: int, *, legacy: bool) -> dict:
    value = {
        "N": index + 2,
        "elements": ["Na", "Cl"],
        "counts": [index + 1, 1],
        "formula": f"Na{index + 1}Cl",
        "reduced_formula": f"Na{index + 1}Cl",
        "charge_bucket": "neutral_plausible",
        "anion_framework": "halide",
        "lattice_system": "cubic",
        "spacegroup_bucket": "sg_195_230",
        "volume_per_atom_bin": "volpa_015_019",
    }
    if legacy:
        value["oxidation_candidates"] = "unknown"
        value["prototype_key"] = MODULE.RICH.prototype_key(value)
    return value


class FreezeFaithfulRichDiagnosticCohortTest(unittest.TestCase):
    def test_r0s_repairs_only_deterministic_schema_fields(self):
        source = [{"sample_idx": 91, "source_ordinal": 7, "plan_state": plan(0, legacy=False)}]
        rows, ledger, report = MODULE.freeze_r0s(source, count=1)
        repaired = rows[0]["plan_state"]
        self.assertEqual(repaired["oxidation_candidates"], "unknown")
        self.assertIsInstance(repaired["prototype_key"], str)
        self.assertNotIn('"oxidation_candidates":null', rows[0]["prompt"])
        self.assertNotIn('"prototype_key":null', rows[0]["prompt"])
        self.assertEqual(rows[0]["sample_idx"], 0)
        self.assertEqual(rows[0]["source_sample_idx"], 91)
        self.assertEqual(ledger[0]["allowed_plan_changes"], ["oxidation_candidates", "prototype_key"])
        self.assertEqual(report["exact_identity_mismatches"], 0)
        self.assertEqual(report["soft_tuple_mismatches"], 0)

    def test_h0_rebuilds_the_historical_canonical_prompt_without_selection(self):
        original_plan = plan(0, legacy=True)
        prompt = MODULE.RICH.build_body_prompt(original_plan).rstrip() + "\n"
        source = [{"sample_idx": 123, "plan_state": original_plan, "prompt": prompt}]
        rows, ledger, report = MODULE.freeze_h0(source, count=1)
        self.assertEqual(rows[0]["sample_idx"], 0)
        self.assertEqual(rows[0]["source_sample_idx"], 123)
        self.assertEqual(rows[0]["prompt"], prompt)
        self.assertTrue(ledger[0]["source_prompt_exact_match"])
        self.assertEqual(report["source_prompt_exact_matches"], 1)

    def test_immutable_writer_refuses_overwrite(self):
        r0_rows = [{"sample_idx": 1, "plan_state": plan(0, legacy=False)}]
        h0_plan = plan(1, legacy=True)
        h0_rows = [{"sample_idx": 2, "plan_state": h0_plan, "prompt": MODULE.RICH.build_body_prompt(h0_plan)}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cohort"
            MODULE.freeze_to_directory(
                r0_source=r0_rows,
                h0_source=h0_rows,
                output_dir=output,
                count=1,
                input_provenance={"r0": {"path": "r0"}, "h0": {"path": "h0"}},
                tokenizer=None,
            )
            self.assertTrue((output / "_SUCCESS").is_file())
            with self.assertRaises(FileExistsError):
                MODULE.freeze_to_directory(
                    r0_source=r0_rows,
                    h0_source=h0_rows,
                    output_dir=output,
                    count=1,
                    input_provenance={},
                    tokenizer=None,
                )


if __name__ == "__main__":
    unittest.main()

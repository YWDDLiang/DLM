import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_spad_species_pointer_data",
    ROOT / "scripts/build_spad_species_pointer_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import SPAD pointer-data builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SPADPointerDataBuilderTest(unittest.TestCase):
    def test_fused_oxidation_variants_collapse_to_plan_composition(self):
        vocabulary = {
            "species": [
                {"id": 0, "atomic_number": 8},
                {"id": 1, "atomic_number": 8},
                {"id": 2, "atomic_number": 11},
            ]
        }
        row = {"species_ids": [0, 1, 2], "count_targets": [1, 2, 1]}
        self.assertEqual(
            MODULE._composition_from_fused(row, vocabulary),
            (("O", 3), ("Na", 1)),
        )

    def test_source_does_not_consume_energy_or_hull_columns(self):
        source = (
            ROOT / "scripts/build_spad_species_pointer_data.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('row["e_above_hull"]', source)
        self.assertNotIn('row["energy"]', source)
        self.assertIn('str(row["cif"])', source)


if __name__ == "__main__":
    unittest.main()

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_listwise_safety_v2",
    ROOT / "scripts" / "build_listwise_safety_v2.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


try:
    import pymatgen  # noqa: F401
except ModuleNotFoundError:
    pymatgen = None


def cif(distance=0.5):
    return f"""data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0 0 0
O1 O {distance} {distance} {distance}
"""


class ListwiseSafetyV2Test(unittest.TestCase):
    def test_composition_identity_parser(self):
        self.assertEqual(MODULE.composition_counts("Li:2|O:1"), {"Li": 2, "O": 1})
        with self.assertRaises(ValueError):
            MODULE.composition_counts("Li:2|Li:1")

    def test_cli_has_no_holdout_or_test_outcome_argument(self):
        options = {
            option
            for action in MODULE.build_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(any("holdout" in option.lower() for option in options))
        self.assertFalse(any("test" in option.lower() for option in options))

    @unittest.skipIf(pymatgen is None, "pymatgen is unavailable")
    def test_raw_structure_safety_distinguishes_overlap(self):
        valid = MODULE.raw_structure_safety(
            {"composition_id": "Li:1|O:1", "cif": cif(0.5)}
        )
        invalid = MODULE.raw_structure_safety(
            {"composition_id": "Li:1|O:1", "cif": cif(0.01)}
        )
        self.assertTrue(valid["raw_direct_joint_valid"])
        self.assertFalse(invalid["raw_structure_valid"])
        self.assertEqual(invalid["raw_missing_reason"], "structure_invalid")

    @unittest.skipIf(pymatgen is None, "pymatgen is unavailable")
    def test_augmentation_selects_lowest_energy_raw_valid_anchor(self):
        answers = ["valid-low", "invalid-lowest", "valid-high"]
        cifs = [cif(0.5), cif(0.01), cif(0.4)]
        energies = [-2.0, -3.0, -1.0]
        candidates = []
        source_index = {}
        for index, (answer, raw_cif, energy) in enumerate(zip(answers, cifs, energies)):
            candidates.append(
                {
                    "candidate_index": index,
                    "answer": answer,
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    "cif_sha256": hashlib.sha256(raw_cif.encode()).hexdigest(),
                    "post_model494_energy_per_atom": energy,
                    "source": "noisy_stream0",
                    "source_ordinal": index,
                }
            )
            source_index[("noisy_stream0", index)] = {"text": answer, "cif": raw_cif}
        group = {
            "schema": "fixture",
            "split": "train",
            "composition_id": "Li:1|O:1",
            "group_weight": 1.0,
            "candidates": candidates,
        }
        augmented, audit = MODULE.augment_groups(
            {"train": [group], "validation": []}, source_index, workers=1
        )
        row = augmented["train"][0]
        self.assertEqual(row["best_valid_candidate_index"], 0)
        self.assertEqual(row["raw_valid_count"], 2)
        self.assertEqual(row["candidates"][1]["safety_rank"], 2)
        self.assertTrue(row["candidates"][0]["is_best_valid_anchor"])
        self.assertEqual(audit["train_raw_valid"], 2)


if __name__ == "__main__":
    unittest.main()

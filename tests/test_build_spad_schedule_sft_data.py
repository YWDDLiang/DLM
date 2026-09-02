import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_spad_schedule_sft_data",
    ROOT / "scripts/build_spad_schedule_sft_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import SPAD SFT builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SPADScheduleSFTBuilderTest(unittest.TestCase):
    def source(self, source_idx):
        return {
            "source_row_idx": source_idx,
            "source_split": "train",
            "prompt": "plan",
            "answer": "body",
            "plan_state": {
                "N": 4,
                "elements": ["O", "Na", "Cl"],
                "counts": [2, 1, 1],
            },
        }

    def test_three_mask_classes_never_mask_N_or_elements(self):
        element_positions = {0, 7, 11, 15, 19}
        modes = set()
        for source_idx in range(3):
            row = MODULE.build_schedule_row(
                self.source(source_idx),
                source_idx=source_idx,
                order=["Cl", "O", "Na"],
                seed=7,
            )
            modes.add(row["spad_mask_class"])
            self.assertTrue(set(row["loss_positions"]) <= set(row["forced_mask_positions"]))
            self.assertFalse(set(row["forced_mask_positions"]) & element_positions)
            self.assertEqual(row["source_answer"], row["answer"])
        self.assertEqual(
            modes,
            {
                "deterministic_random_geometry",
                "program_predictor",
                "suffix_visible_anchor_correction",
            },
        )

    def test_missing_pointer_semantics_uses_disclosed_canonical_program(self):
        row = MODULE.build_schedule_row(
            self.source(4), source_idx=4, order=None, seed=7
        )
        self.assertEqual(row["species_program"], ["O", "Na", "Cl"])
        self.assertEqual(
            row["species_program_source"], "canonical_missing_pointer_semantics"
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_full_mp20_pointer_programs",
    ROOT / "src/scripts/export_full_mp20_pointer_programs.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import full-MP20 pointer exporter")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullMP20PointerProgramsTest(unittest.TestCase):
    def sft_row(self, source_idx, *, fallback=False):
        elements = ["O", "Na", "Cl"]
        return {
            "source_row_idx": source_idx,
            "prompt": f"plan-{source_idx}",
            "answer": f"body-{source_idx}",
            "plan_state": {"N": 4, "elements": elements, "counts": [2, 1, 1]},
            "species_program": elements,
            "species_program_source": (
                MODULE.FALLBACK_SOURCE if fallback else "contact_tree_teacher"
            ),
        }

    def pointer_row(self, source_idx, *, atomic_numbers=None):
        return {
            "source_row_idx": source_idx,
            "canonical_atomic_numbers": atomic_numbers or [8, 11, 17],
            "canonical_element_counts": [2, 1, 1],
        }

    def test_join_decodes_covered_rows_and_preserves_declared_fallback(self):
        sft = [
            self.sft_row(0),
            self.sft_row(1, fallback=True),
            self.sft_row(2),
        ]
        pointer = [self.pointer_row(0), self.pointer_row(2)]
        output, manifest = MODULE.join_full_mp20_programs(
            sft,
            pointer,
            {0: [2, 0, 1], 2: [1, 2, 0]},
            expected_sft_rows=3,
            expected_pointer_rows=2,
        )

        self.assertEqual([row["sample_idx"] for row in output], [0, 1, 2])
        self.assertEqual([row["source_row_idx"] for row in output], [0, 1, 2])
        self.assertEqual(output[0]["species_program"], ["Cl", "O", "Na"])
        self.assertEqual(output[0]["species_program_source"], MODULE.POINTER_SOURCE)
        self.assertEqual(output[1]["species_program"], ["O", "Na", "Cl"])
        self.assertEqual(output[1]["species_program_indices"], [0, 1, 2])
        self.assertEqual(output[1]["species_program_source"], MODULE.FALLBACK_SOURCE)
        self.assertEqual(output[1]["prompt"], "plan-1")
        self.assertEqual(output[1]["teacher_answer"], "body-1")
        self.assertFalse(any(row["outcomes_read"] for row in output))
        self.assertEqual(manifest["canonical_missing_pointer_source_row_indices"], [1])
        self.assertEqual(manifest["pointer_decoded_rows"], 2)
        self.assertEqual(manifest["canonical_missing_pointer_rows"], 1)
        self.assertFalse(manifest["typed_transcript_fabricated_for_missing_rows"])

    def test_rejects_nonpermutation_and_pointer_plan_mismatch(self):
        sft = [self.sft_row(0), self.sft_row(1, fallback=True)]
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            MODULE.join_full_mp20_programs(
                sft,
                [self.pointer_row(0)],
                {0: [0, 0, 2]},
                expected_sft_rows=2,
                expected_pointer_rows=1,
            )
        with self.assertRaisesRegex(ValueError, "Plan differ"):
            MODULE.join_full_mp20_programs(
                sft,
                [self.pointer_row(0, atomic_numbers=[11, 8, 17])],
                {0: [0, 1, 2]},
                expected_sft_rows=2,
                expected_pointer_rows=1,
            )

    def test_missing_pointer_row_must_be_declared_canonical_in_sft(self):
        with self.assertRaisesRegex(ValueError, "declared canonical fallback"):
            MODULE.join_full_mp20_programs(
                [self.sft_row(0), self.sft_row(1)],
                [self.pointer_row(0)],
                {0: [0, 1, 2]},
                expected_sft_rows=2,
                expected_pointer_rows=1,
            )

    def test_manifests_must_account_for_the_exact_missing_set_size(self):
        sft_manifest = {
            "schema": MODULE.SFT_MANIFEST_SCHEMA,
            "splits": {
                "train": {
                    "rows": 3,
                    "program_sources": {MODULE.FALLBACK_SOURCE: 1},
                }
            },
        }
        pointer_manifest = {
            "schema": MODULE.POINTER_MANIFEST_SCHEMA,
            "splits": {"train": {"rows": 2}},
        }
        MODULE.validate_input_manifests(
            sft_manifest,
            pointer_manifest,
            expected_sft_rows=3,
            expected_pointer_rows=2,
        )
        sft_manifest["splits"]["train"]["program_sources"][
            MODULE.FALLBACK_SOURCE
        ] = 0
        with self.assertRaisesRegex(ValueError, "fallback count"):
            MODULE.validate_input_manifests(
                sft_manifest,
                pointer_manifest,
                expected_sft_rows=3,
                expected_pointer_rows=2,
            )


if __name__ == "__main__":
    unittest.main()

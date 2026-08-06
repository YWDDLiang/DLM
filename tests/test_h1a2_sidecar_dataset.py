import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.h1a2_sidecar_dataset import (
    H1A2_SIDECAR_DATASET_VERSION,
    H1A2SidecarBuildError,
    build_h1a2_sidecar_dataset,
    build_h1a2_sidecar_record,
    model_visible_text,
)
from crystal_dlm.plangraph_v1 import validate_plangraph
from crystal_dlm.planned_corruption import plan_condition_sha256
from crystal_dlm.r5_plan_state import build_body_prompt, plan_state_from_arrays


class H1A2SidecarDatasetTests(unittest.TestCase):
    def make_source(self):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(
            arrays,
            metadata={
                "material_id": "mp-secret",
                "spacegroup.number": 194,
                "e_above_hull": -99.0,
            },
        )
        prompt = build_body_prompt(plan)
        return {
            "task": "r5_exact_dynamic_body",
            "representation": "dynamic_v1",
            "r5_representation": "r5_exact_dynamic_v1",
            "prompt": prompt,
            "answer": answer,
            "text": prompt.rstrip() + "\n" + answer,
            "loss_profile": "fixed_slot",
            "sample_weight": 1.0,
            "num_atoms": 3,
            "answer_semantic_length": 19,
            "answer_token_count": 19,
            "plan_state": plan,
            "r5_plan_state": plan,
            "metadata": {
                "material_id": "mp-secret",
                "formation_energy_per_atom": -123.0,
                "e_above_hull": -99.0,
                "sun": True,
            },
        }

    def test_record_keeps_model_visible_bytes_and_strips_source_metadata(self):
        source = self.make_source()
        record = build_h1a2_sidecar_record(source)

        self.assertEqual(record["prompt"], source["prompt"])
        self.assertEqual(record["answer"], source["answer"])
        self.assertEqual(record["text"], source["text"])
        self.assertEqual(record["text"], model_visible_text(source))
        self.assertEqual(record["sample_weight"], 1.0)
        self.assertTrue(validate_plangraph(record["plangraph"]).valid)
        self.assertEqual(
            record["plan_condition_sha256"],
            plan_condition_sha256(
                prompt=record["prompt"],
                graph=record["plangraph"],
            ),
        )

        encoded = json.dumps(record, sort_keys=True).lower()
        for forbidden in (
            "mp-secret",
            "formation_energy",
            "e_above_hull",
            '"sun"',
            '"metadata"',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_record_refuses_non_h1a2_sample_weight(self):
        source = self.make_source()
        source["sample_weight"] = 2.0
        with self.assertRaises(Exception):
            build_h1a2_sidecar_record(source)

    def test_record_refuses_prompt_answer_boundary_change(self):
        source = self.make_source()
        source["text"] = str(source["text"]) + " "
        with self.assertRaises(Exception):
            build_h1a2_sidecar_record(source)

    def test_atomic_builder_publishes_full_denominator_dataset(self):
        source = self.make_source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "train.jsonl"
            source_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            vocab = root / "vocab_tokens.txt"
            vocab.write_text("<N_003>\n", encoding="utf-8")
            output = root / "published"

            manifest = build_h1a2_sidecar_dataset(
                split_inputs={"train": source_path},
                output_dir=output,
                vocab_file=vocab,
                project_root=root,
            )

            self.assertEqual(
                manifest["dataset_version"],
                H1A2_SIDECAR_DATASET_VERSION,
            )
            self.assertEqual(manifest["converted_rows"], 1)
            self.assertTrue(manifest["prompt_answer_byte_identity"])
            self.assertEqual(
                manifest["source_plan_provenance"],
                "structure_derived_teacher_plan_state",
            )
            self.assertFalse(manifest["model_proposed_plan"])
            self.assertFalse(
                manifest["eligible_as_end_to_end_planner_evidence"]
            )
            self.assertTrue((output / "train.jsonl").is_file())
            self.assertTrue((output / "row_ledger/train.jsonl").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "_SUCCESS").is_file())
            published = json.loads(
                (output / "train.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(published["text"], source["text"])

            with self.assertRaises(H1A2SidecarBuildError):
                build_h1a2_sidecar_dataset(
                    split_inputs={"train": source_path},
                    output_dir=output,
                    vocab_file=vocab,
                    project_root=root,
                )

    def test_builder_rejects_cross_split_duplicate(self):
        source = self.make_source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            payload = json.dumps(source, ensure_ascii=False) + "\n"
            train.write_text(payload, encoding="utf-8")
            val.write_text(payload, encoding="utf-8")
            vocab = root / "vocab_tokens.txt"
            vocab.write_text("<N_003>\n", encoding="utf-8")
            with self.assertRaises(H1A2SidecarBuildError) as context:
                build_h1a2_sidecar_dataset(
                    split_inputs={"train": train, "val": val},
                    output_dir=root / "published",
                    vocab_file=vocab,
                    project_root=root,
                )
            self.assertTrue(
                context.exception.report["cross_split_overlaps"]
            )


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.h1_readonly_guard import H1ReadOnlyViolation
from crystal_dlm.plangraph_dataset import (
    PlanGraphDatasetBuildError,
    build_plangraph_dataset,
    sha256_file,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class PlanGraphDatasetTests(unittest.TestCase):
    def make_source(self, *, shift: float = 0.0):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1 + shift, 3.1, 5.2],
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
                "material_id": f"mp-secret-{shift}",
                "spacegroup.number": 194,
                "e_above_hull": 123.0 + shift,
            },
        )
        return {
            "representation": "dynamic_v1",
            "plan_state": plan,
            "answer": answer,
            "prompt": "formation_energy=-999; S.U.N.=true; MP_API_KEY=secret",
            "metadata": {
                "material_id": f"mp-secret-{shift}",
                "formation_energy": -999.0,
                "chgnet_score": 1.0,
            },
        }

    @staticmethod
    def write_jsonl(path: Path, rows) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_atomic_build_freezes_rows_and_excludes_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            self.write_jsonl(
                source_dir / "train.jsonl",
                [self.make_source(shift=0.0), self.make_source(shift=0.1)],
            )
            self.write_jsonl(
                source_dir / "val.jsonl",
                [self.make_source(shift=0.2)],
            )
            vocab = source_dir / "vocab_tokens.txt"
            vocab.write_text("<N_3>\n<E_Li>\n<E_O>\n", encoding="utf-8")
            output = root / "published"

            report = build_plangraph_dataset(
                split_inputs={
                    "train": source_dir / "train.jsonl",
                    "val": source_dir / "val.jsonl",
                },
                output_dir=output,
                vocab_file=vocab,
                project_root=root,
            )

            self.assertTrue(report["published"])
            self.assertEqual(report["total_rows"], 3)
            self.assertEqual(report["failed_rows"], 0)
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(
                sha256_file(output / "manifest.json"),
                report["manifest_sha256"],
            )
            self.assertEqual(
                report["fixed_validation_panel"]["row_count"],
                1,
            )
            train_body = [
                json.loads(line)
                for line in (output / "body/train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            ledger = [
                json.loads(line)
                for line in (output / "row_ledger/train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([item["ordinal"] for item in ledger], [0, 1])
            self.assertEqual(len(train_body), 2)
            self.assertEqual(
                [item["training_pair_sha256"] for item in train_body],
                [item["training_pair_sha256"] for item in ledger],
            )
            all_outputs = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file()
            ).lower()
            for forbidden in (
                "mp-secret",
                "formation_energy",
                "e_above_hull",
                "chgnet_score",
                "mp_api_key",
                '"material_id"',
            ):
                self.assertNotIn(forbidden, all_outputs)

            with self.assertRaises(PlanGraphDatasetBuildError):
                build_plangraph_dataset(
                    split_inputs={"train": source_dir / "train.jsonl"},
                    output_dir=output,
                    vocab_file=vocab,
                    project_root=root,
                )

    def test_any_failed_row_keeps_denominator_and_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "train.jsonl"
            source.write_text(
                json.dumps(self.make_source()) + "\n" + "{not-json}\n",
                encoding="utf-8",
            )
            vocab = root / "vocab_tokens.txt"
            vocab.write_text("<N_3>\n", encoding="utf-8")
            output = root / "must-not-exist"

            with self.assertRaises(PlanGraphDatasetBuildError) as caught:
                build_plangraph_dataset(
                    split_inputs={"train": source},
                    output_dir=output,
                    vocab_file=vocab,
                    project_root=root,
                )

            self.assertFalse(output.exists())
            report = caught.exception.report
            self.assertEqual(report["total_rows"], 2)
            self.assertEqual(report["converted_rows"], 1)
            self.assertEqual(report["failed_rows"], 1)
            self.assertFalse(report["publication_gate_passed"])
            self.assertEqual(
                report["splits"]["train"]["failure_categories"],
                {"json_decode": 1},
            )

    def test_cross_split_identity_overlap_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = self.make_source()
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            self.write_jsonl(train, [row])
            self.write_jsonl(val, [row])
            vocab = root / "vocab_tokens.txt"
            vocab.write_text("<N_3>\n", encoding="utf-8")
            output = root / "must-not-exist"

            with self.assertRaises(PlanGraphDatasetBuildError) as caught:
                build_plangraph_dataset(
                    split_inputs={"train": train, "val": val},
                    output_dir=output,
                    vocab_file=vocab,
                    project_root=root,
                )

            self.assertFalse(output.exists())
            overlaps = caught.exception.report["cross_split_identity_overlaps"]
            self.assertEqual(len(overlaps), 1)
            self.assertEqual(overlaps[0]["count"], 1)

    def test_frozen_h1_output_is_refused_before_staging(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "train.jsonl"
            self.write_jsonl(source, [self.make_source()])
            vocab = temp / "vocab_tokens.txt"
            vocab.write_text("<N_3>\n", encoding="utf-8")
            with self.assertRaises(H1ReadOnlyViolation):
                build_plangraph_dataset(
                    split_inputs={"train": source},
                    output_dir=(
                        root
                        / "runs/20260729_h1a2c_jointchem_v1/new_data"
                    ),
                    vocab_file=vocab,
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()

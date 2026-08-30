import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_c3fd_native_sft_data",
    ROOT / "scripts" / "build_c3fd_native_sft_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import build_c3fd_native_sft_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class BuildC3FDNativeSFTDataTest(unittest.TestCase):
    def plan(self, split):
        if split == "train":
            return {
                "N": 3,
                "elements": ["Li", "O"],
                "counts": [2, 1],
                "formula": "Li2O",
                "anion_framework": "oxide",
                "charge_bucket": "neutral_plausible",
                "lattice_system": "cubic",
                "spacegroup_bucket": "sg_195_230",
                "volume_per_atom_bin": "volpa_015_019",
                "prototype_key": "must-not-leak",
                "oxidation_candidates": "unknown",
                "metadata": {"e_above_hull": 0.0},
            }
        return {
            "N": 2,
            "elements": ["Na", "Cl"],
            "counts": [1, 1],
            "formula": "NaCl",
            "anion_framework": "halide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "hexagonal",
            "spacegroup_bucket": "sg_143_167",
            "volume_per_atom_bin": "volpa_020_024",
            "prototype_key": "must-not-leak",
            "oxidation_candidates": "unknown",
            "target_stability": "source-only",
        }

    def predicted(self, split):
        if split == "train":
            return {
                "lattice_system": "tetragonal",
                "spacegroup_bucket": "sg_075_142",
                "volume_per_atom_bin": "volpa_025_029",
            }
        return {
            "lattice_system": "orthorhombic",
            "spacegroup_bucket": "sg_016_074",
            "volume_per_atom_bin": "volpa_030_039",
        }

    def make_inputs(self, root):
        source = root / "source"
        semantic = root / "semantic"
        predicted = root / "predicted"
        for directory in (source, semantic, predicted):
            directory.mkdir()
        (source / "vocab_tokens.txt").write_text(
            "<N_002>\n<N_003>\n", encoding="utf-8"
        )
        vocabulary = {
            "schema": "h1a2_c3fd_semantic_vocabulary_v1",
            "species": [
                {"id": 0, "atomic_number": 3, "oxidation_state": 1},
                {"id": 1, "atomic_number": 8, "oxidation_state": -2},
                {"id": 2, "atomic_number": 11, "oxidation_state": 1},
                {"id": 3, "atomic_number": 17, "oxidation_state": -1},
            ],
        }
        (semantic / "vocabulary.json").write_text(
            json.dumps(vocabulary), encoding="utf-8"
        )
        for split in ("train", "val"):
            plan = self.plan(split)
            answer = "<N_003><LATTICE>body" if split == "train" else "<N_002><LATTICE>body"
            write_jsonl(
                source / f"{split}.jsonl",
                [
                    {
                        "source_split": split,
                        "source_row_idx": 0,
                        "c3fd_certificate_source_row_idx": 0,
                        "prompt": "legacy source prompt",
                        "answer": answer,
                        "plan_state": plan,
                        "sample_weight": 2.0,
                        "loss_profile": "dynamic_crystal",
                        "e_above_hull": 0.0,
                    }
                ],
            )
            labels = [0, 1] if split == "train" else [2, 3]
            counts = [2, 1] if split == "train" else [1, 1]
            write_jsonl(
                semantic / f"{split}.jsonl",
                [
                    {
                        "schema": "h1a2_c3fd_semantic_row_v1",
                        "split": split,
                        "source_row_idx": 0,
                        "composition_supervision": True,
                        "plan_state": plan,
                        "species_labels": labels,
                        "count_targets": counts,
                        "soft_labels": {"ignored": 1},
                    }
                ],
            )
            write_jsonl(
                predicted / f"{split}.jsonl",
                [
                    {
                        "split": split,
                        "source_row_idx": 0,
                        "predicted_soft_fields": self.predicted(split),
                    }
                ],
            )
        return source, semantic, predicted

    def build(self, root):
        source, semantic, predicted = self.make_inputs(root)
        output = root / "output"
        manifest = MODULE.build_dataset(
            input_dir=source,
            semantic_dir=semantic,
            predicted_soft_dir=predicted,
            output_dir=output,
        )
        return source, semantic, predicted, output, manifest

    def test_roundtrip_and_four_views_share_exact_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            _source, _semantic, _predicted, output, manifest = self.build(Path(temp))
            rows = read_jsonl(output / "train.jsonl")
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["view"] for row in rows}, set(MODULE.VIEWS))
            self.assertEqual(len({row["answer"] for row in rows}), 1)
            self.assertEqual(len({row["answer_sha256"] for row in rows}), 1)
            self.assertAlmostEqual(sum(row["sample_weight"] for row in rows), 2.0)
            for name in ("teacher-native", "predicted-native"):
                row = next(value for value in rows if value["view"] == name)
                parsed = MODULE.parse_native_plan_line(row["native_plan_line"])
                self.assertEqual(parsed["N"], 3)
                self.assertEqual(parsed["elements"], ["Li", "O"])
                self.assertEqual(parsed["counts"], [2, 1])
                self.assertTrue(parsed["charge_bucket_match"])
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertTrue((output / "_SUCCESS").is_file())
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(MODULE.sha256_file(output / name), digest)
            self.assertEqual(manifest["chemsys_overlap"], {"train__val": 0})

    def test_predicted_native_changes_only_three_soft_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            _source, _semantic, _predicted, output, _manifest = self.build(Path(temp))
            rows = {row["view"]: row for row in read_jsonl(output / "train.jsonl")}
            teacher = MODULE._line_fields(rows["teacher-native"]["native_plan_line"])
            predicted = MODULE._line_fields(rows["predicted-native"]["native_plan_line"])
            changed = {key for key in teacher if teacher[key] != predicted[key]}
            self.assertEqual(changed, {"LS", "SG", "VP"})
            for key in set(teacher) - {"LS", "SG", "VP"}:
                self.assertEqual(teacher[key], predicted[key])

    def test_prompts_and_payloads_copy_no_forbidden_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            _source, _semantic, _predicted, output, _manifest = self.build(Path(temp))
            for split in ("train", "val"):
                for row in read_jsonl(output / f"{split}.jsonl"):
                    prompt = row["prompt"].lower()
                    for fragment in MODULE.FORBIDDEN_PROMPT_FRAGMENTS:
                        self.assertNotIn(fragment, prompt)
                    payload = json.dumps(row, sort_keys=True).lower()
                    self.assertNotIn("e_above_hull", payload)
                    self.assertNotIn("target_stability", payload)
                    self.assertNotIn("prototype_key", payload)
                    self.assertNotIn("oxidation_candidates", payload)
            masked = next(
                row
                for row in read_jsonl(output / "train.jsonl")
                if row["view"] == "soft-masked"
            )
            self.assertIn("LS=<SOFT_MASK>", masked["native_plan_line"])
            self.assertIn("SG=<SOFT_MASK>", masked["native_plan_line"])
            self.assertIn("VP=<SOFT_MASK>", masked["native_plan_line"])

    def test_missing_or_misaligned_predicted_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            write_jsonl(predicted / "val.jsonl", [])
            with self.assertRaisesRegex(ValueError, "predicted split length changed"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "missing-output",
                )
            self.assertFalse((root / "missing-output").exists())

            write_jsonl(
                predicted / "val.jsonl",
                [
                    {
                        "split": "val",
                        "source_row_idx": 1,
                        "predicted_soft_fields": self.predicted("val"),
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "source_row_idx changed"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "misaligned-output",
                )
            self.assertFalse((root / "misaligned-output").exists())

    def test_missing_predictions_cannot_fall_back_to_teacher_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, _predicted = self.make_inputs(root)
            with self.assertRaisesRegex(ValueError, "predicted soft fields are missing"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    output_dir=root / "no-predictions",
                )
            self.assertFalse((root / "no-predictions").exists())

            for split in ("train", "val"):
                rows = read_jsonl(semantic / f"{split}.jsonl")
                rows[0]["frozen_predicted_soft_fields"] = self.predicted(split)
                write_jsonl(semantic / f"{split}.jsonl", rows)
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                output_dir=root / "inline-predictions",
            )
            self.assertIsNone(manifest["predicted_soft_dir"])
            self.assertTrue((root / "inline-predictions" / "_SUCCESS").is_file())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted, output, _manifest = self.build(root)
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=output,
                )
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_valence_mismatch_and_chemsys_leakage_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            semantic_rows = read_jsonl(semantic / "val.jsonl")
            semantic_rows[0]["count_targets"] = [2, 1]
            write_jsonl(semantic / "val.jsonl", semantic_rows)
            with self.assertRaisesRegex(ValueError, "valence species changed exact composition"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "bad-valence",
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            train_source = read_jsonl(source / "train.jsonl")[0]
            train_semantic = read_jsonl(semantic / "train.jsonl")[0]
            val_source = dict(train_source)
            val_source["source_split"] = "val"
            val_semantic = dict(train_semantic)
            val_semantic["split"] = "val"
            write_jsonl(source / "val.jsonl", [val_source])
            write_jsonl(semantic / "val.jsonl", [val_semantic])
            with self.assertRaisesRegex(ValueError, "chemsys-held-out split leakage"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "leaked-split",
                )
            self.assertFalse((root / "leaked-split").exists())


if __name__ == "__main__":
    unittest.main()

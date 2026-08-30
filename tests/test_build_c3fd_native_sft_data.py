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

    def checkpoint_prediction(self, split, checkpoint):
        if checkpoint == "seed17":
            return self.predicted(split)
        if split == "train":
            return {
                "lattice_system": "monoclinic",
                "spacegroup_bucket": "sg_003_015",
                "volume_per_atom_bin": "volpa_040_049",
            }
        return {
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "<UNKNOWN>",
        }

    def refresh_formal_manifest_hashes(self, predicted):
        manifest_path = predicted / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split in ("train", "val"):
            digest = MODULE.sha256_file(predicted / f"{split}.jsonl")
            manifest["splits"][split]["output_sha256"] = digest
            manifest["outputs"][f"{split}.jsonl"] = digest
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        frozen_names = ("manifest.json", "train.jsonl", "val.jsonl")
        checksum_path = predicted / "SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{MODULE.sha256_file(predicted / name)}  {name}\n"
                for name in sorted(frozen_names)
            ),
            encoding="utf-8",
        )
        (predicted / "_SUCCESS").write_text(
            json.dumps(
                {
                    "manifest_sha256": MODULE.sha256_file(manifest_path),
                    "sha256sums_sha256": MODULE.sha256_file(checksum_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def make_formal_inputs(self, root):
        source, semantic, predicted = self.make_inputs(root)
        for split in ("train", "val"):
            plan = self.plan(split)
            by_checkpoint = {}
            for checkpoint in MODULE.FORMAL_CHECKPOINT_ORDER:
                by_checkpoint[checkpoint] = {
                    field: {
                        "prediction": value,
                        "confidence": 0.9 if checkpoint == "seed17" else 0.8,
                    }
                    for field, value in self.checkpoint_prediction(
                        split, checkpoint
                    ).items()
                }
            write_jsonl(
                predicted / f"{split}.jsonl",
                [
                    {
                        "schema": MODULE.FORMAL_PREDICTION_ROW_SCHEMA,
                        "split": split,
                        "source_row_idx": 0,
                        "N": plan["N"],
                        "elements": plan["elements"],
                        "counts": plan["counts"],
                        "predictions_by_checkpoint": by_checkpoint,
                    }
                ],
            )
        manifest = {
            "schema": MODULE.FORMAL_PREDICTION_MANIFEST_SCHEMA,
            "outcomes_read": False,
            "selection": "none",
            "all_frozen_checkpoints_preserved": True,
            "expected_checkpoint_seeds": [17, 18],
            "checkpoint_order": ["seed17", "seed18"],
            "prediction_fields": list(MODULE.SOFT_FIELDS),
            "checkpoints": [
                {"name": "seed17", "seed": 17},
                {"name": "seed18", "seed": 18},
            ],
            "splits": {
                split: {
                    "output_sha256": "pending",
                    "predictions": {"seed17": {}, "seed18": {}},
                }
                for split in ("train", "val")
            },
            "outputs": {
                "train.jsonl": "pending",
                "val.jsonl": "pending",
            },
        }
        (predicted / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.refresh_formal_manifest_hashes(predicted)
        return source, semantic, predicted

    def build(self, root):
        source, semantic, predicted = self.make_inputs(root)
        output = root / "output"
        manifest = MODULE.build_dataset(
            input_dir=source,
            semantic_dir=semantic,
            predicted_soft_dir=predicted,
            output_dir=output,
            allow_legacy_single_prediction_development=True,
        )
        return source, semantic, predicted, output, manifest

    def test_formal_export_preserves_both_checkpoint_views_without_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            output = root / "formal-output"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
            )
            rows = read_jsonl(output / "train.jsonl")
            expected_views = [
                "teacher-native",
                "predicted-native-seed17",
                "predicted-native-seed18",
                "soft-masked",
            ]
            self.assertEqual([row["view"] for row in rows], expected_views)
            self.assertEqual(len({row["answer"] for row in rows}), 1)
            self.assertEqual(len({row["answer_sha256"] for row in rows}), 1)
            self.assertAlmostEqual(sum(row["sample_weight"] for row in rows), 2.0)
            self.assertTrue(all(row["sample_weight"] == 0.5 for row in rows))
            predicted_rows = [
                row for row in rows if row["view"].startswith("predicted-native-")
            ]
            self.assertEqual(
                [row["prediction_checkpoint"] for row in predicted_rows],
                ["seed17", "seed18"],
            )
            self.assertFalse(any("selected" in key for row in rows for key in row))
            teacher = MODULE._line_fields(rows[0]["native_plan_line"])
            for row in predicted_rows:
                predicted_fields = MODULE._line_fields(row["native_plan_line"])
                changed = {
                    key
                    for key in teacher
                    if teacher[key] != predicted_fields[key]
                }
                self.assertTrue(changed)
                self.assertLessEqual(
                    changed,
                    {"lattice_system", "spacegroup_bucket", "volume_per_atom_bin"},
                )
            self.assertEqual(manifest["prediction_mode"], "formal-multi-checkpoint")
            self.assertEqual(
                manifest["prediction_checkpoint_order"], ["seed17", "seed18"]
            )
            self.assertEqual(
                manifest["predicted_view_names"],
                ["predicted-native-seed17", "predicted-native-seed18"],
            )
            self.assertEqual(manifest["prediction_selection"], "none")
            self.assertFalse(manifest["legacy_single_prediction_development"])
            val_seed18 = next(
                row
                for row in read_jsonl(output / "val.jsonl")
                if row["view"] == "predicted-native-seed18"
            )
            self.assertEqual(
                json.loads(val_seed18["native_plan_line"])["volume_per_atom_bin"],
                "<SOFT_MASK>",
            )

    def test_formal_sft_teacher_only_uses_one_rich_json_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, _predicted = self.make_formal_inputs(root)
            output = root / "teacher-only"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                output_dir=output,
                teacher_only=True,
            )
            self.assertEqual(manifest["prediction_mode"], "teacher-only")
            self.assertEqual(manifest["prediction_checkpoint_order"], [])
            self.assertEqual(manifest["views"], ["teacher-native"])
            self.assertEqual(manifest["training_view_mode"], "teacher-rich-json-only")
            for split in ("train", "val"):
                rows = read_jsonl(output / f"{split}.jsonl")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["view"], "teacher-native")
                self.assertEqual(rows[0]["prompt_schema"], "C3FD_NATIVE_PLAN_V2")
                self.assertEqual(rows[0]["sample_weight"], 2.0)
                self.assertNotIn("prediction_checkpoint", rows[0])
            self.assertTrue(all(manifest["gate"].values()))

    def test_roundtrip_and_legacy_development_views_share_exact_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            _source, _semantic, _predicted, output, manifest = self.build(Path(temp))
            rows = read_jsonl(output / "train.jsonl")
            self.assertEqual(len(rows), 3)
            expected_views = set(
                MODULE.expanded_view_names(
                    (MODULE.LEGACY_DEVELOPMENT_CHECKPOINT,)
                )
            )
            self.assertEqual({row["view"] for row in rows}, expected_views)
            self.assertEqual(len({row["answer"] for row in rows}), 1)
            self.assertEqual(len({row["answer_sha256"] for row in rows}), 1)
            self.assertAlmostEqual(sum(row["sample_weight"] for row in rows), 2.0)
            for name in (
                "teacher-native",
                "predicted-native-development-single",
            ):
                row = next(value for value in rows if value["view"] == name)
                parsed = MODULE.parse_native_plan_line(row["native_plan_line"])
                self.assertEqual(parsed["N"], 3)
                self.assertEqual(parsed["elements"], ["Li", "O"])
                self.assertEqual(parsed["counts"], [2, 1])
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertTrue((output / "_SUCCESS").is_file())
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(MODULE.sha256_file(output / name), digest)
            self.assertEqual(manifest["chemsys_overlap"], {"train__val": 0})
            self.assertEqual(
                manifest["predicted_view_names"],
                ["predicted-native-development-single"],
            )
            self.assertTrue(manifest["legacy_single_prediction_development"])

    def test_predicted_native_changes_only_three_soft_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            output = root / "formal-soft-output"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
            )
            rows = {row["view"]: row for row in read_jsonl(output / "train.jsonl")}
            teacher = MODULE._line_fields(rows["teacher-native"]["native_plan_line"])
            for view in ("predicted-native-seed17", "predicted-native-seed18"):
                predicted_fields = MODULE._line_fields(rows[view]["native_plan_line"])
                changed = {
                    key for key in teacher if teacher[key] != predicted_fields[key]
                }
                soft = {
                    "lattice_system",
                    "spacegroup_bucket",
                    "volume_per_atom_bin",
                }
                self.assertEqual(changed, soft)
                for key in set(teacher) - soft:
                    self.assertEqual(teacher[key], predicted_fields[key])

    def test_prompts_and_payloads_copy_no_forbidden_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            output = root / "formal-no-leak-output"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
            )
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
            masked_payload = json.loads(masked["native_plan_line"])
            self.assertEqual(masked_payload["lattice_system"], "<SOFT_MASK>")
            self.assertEqual(masked_payload["spacegroup_bucket"], "<SOFT_MASK>")
            self.assertEqual(masked_payload["volume_per_atom_bin"], "<SOFT_MASK>")

    def test_missing_or_misaligned_predicted_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            write_jsonl(predicted / "val.jsonl", [])
            with self.assertRaisesRegex(ValueError, "missing source_row_idx"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "missing-output",
                    allow_legacy_single_prediction_development=True,
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
            with self.assertRaisesRegex(ValueError, "missing source_row_idx"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "misaligned-output",
                    allow_legacy_single_prediction_development=True,
                )
            self.assertFalse((root / "misaligned-output").exists())

    def test_filtered_source_rows_join_full_semantic_rows_by_source_idx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            for split in ("train", "val"):
                source_rows = read_jsonl(source / f"{split}.jsonl")
                source_rows[0]["source_row_idx"] = 16
                source_rows[0]["c3fd_certificate_source_row_idx"] = 16
                write_jsonl(source / f"{split}.jsonl", source_rows)

                semantic_rows = read_jsonl(semantic / f"{split}.jsonl")
                semantic_rows[0]["source_row_idx"] = 16
                extra_semantic = dict(semantic_rows[0])
                extra_semantic["source_row_idx"] = 15
                write_jsonl(
                    semantic / f"{split}.jsonl",
                    [extra_semantic, semantic_rows[0]],
                )

                predicted_rows = read_jsonl(predicted / f"{split}.jsonl")
                predicted_rows[0]["source_row_idx"] = 16
                extra_predicted = dict(predicted_rows[0])
                extra_predicted["source_row_idx"] = 15
                write_jsonl(
                    predicted / f"{split}.jsonl",
                    [extra_predicted, predicted_rows[0]],
                )
            output = root / "keyed-join"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
                allow_legacy_single_prediction_development=True,
            )
            self.assertEqual(
                {row["source_row_idx"] for row in read_jsonl(output / "train.jsonl")},
                {16},
            )
            self.assertEqual(manifest["splits"]["train"]["unused_semantic_rows"], 1)
            self.assertEqual(manifest["splits"]["train"]["unused_predicted_rows"], 1)

    def test_formal_checkpoint_support_and_order_fail_closed(self):
        for case in ("missing", "extra", "disagreeing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, semantic, predicted = self.make_formal_inputs(root)
                rows = read_jsonl(predicted / "val.jsonl")
                checkpoints = rows[0]["predictions_by_checkpoint"]
                if case == "missing":
                    checkpoints.pop("seed18")
                elif case == "extra":
                    checkpoints["seed19"] = dict(checkpoints["seed18"])
                else:
                    checkpoints["seed19"] = checkpoints.pop("seed18")
                write_jsonl(predicted / "val.jsonl", rows)
                self.refresh_formal_manifest_hashes(predicted)
                with self.assertRaisesRegex(
                    ValueError,
                    "checkpoint support/order disagrees with manifest",
                ):
                    MODULE.build_dataset(
                        input_dir=source,
                        semantic_dir=semantic,
                        predicted_soft_dir=predicted,
                        output_dir=root / f"formal-{case}",
                    )
                self.assertFalse((root / f"formal-{case}").exists())

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            manifest_path = predicted / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoint_order"] = ["seed18", "seed17"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.refresh_formal_manifest_hashes(predicted)
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint support/order must be exactly seed17,seed18",
            ):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "formal-order",
                )

    def test_missing_predictions_cannot_fall_back_to_teacher_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            with self.assertRaisesRegex(ValueError, "development-only"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=root / "legacy-without-development-flag",
                )
            with self.assertRaisesRegex(ValueError, "predicted soft fields are missing"):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    output_dir=root / "no-predictions",
                    allow_legacy_single_prediction_development=True,
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
                allow_legacy_single_prediction_development=True,
            )
            self.assertIsNone(manifest["predicted_soft_dir"])
            self.assertTrue((root / "inline-predictions" / "_SUCCESS").is_file())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            output = root / "formal-immutable-output"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
            )
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                MODULE.build_dataset(
                    input_dir=source,
                    semantic_dir=semantic,
                    predicted_soft_dir=predicted,
                    output_dir=output,
                )
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_auxiliary_valence_is_ignored_and_chemsys_leakage_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_inputs(root)
            semantic_rows = read_jsonl(semantic / "val.jsonl")
            semantic_rows[0]["count_targets"] = [2, 1]
            write_jsonl(semantic / "val.jsonl", semantic_rows)
            output = root / "auxiliary-valence-ignored"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
                allow_legacy_single_prediction_development=True,
            )
            self.assertTrue((output / "_SUCCESS").is_file())

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
                    allow_legacy_single_prediction_development=True,
                )
            self.assertFalse((root / "leaked-split").exists())

    def test_mp20_standard_split_discloses_chemsys_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            train_source = read_jsonl(source / "train.jsonl")[0]
            train_semantic = read_jsonl(semantic / "train.jsonl")[0]
            train_predicted = read_jsonl(predicted / "train.jsonl")[0]
            val_source = dict(train_source)
            val_source["source_split"] = "val"
            val_semantic = dict(train_semantic)
            val_semantic["split"] = "val"
            val_predicted = dict(train_predicted)
            val_predicted["split"] = "val"
            write_jsonl(source / "val.jsonl", [val_source])
            write_jsonl(semantic / "val.jsonl", [val_semantic])
            write_jsonl(predicted / "val.jsonl", [val_predicted])
            self.refresh_formal_manifest_hashes(predicted)

            output = root / "mp20-standard"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
                split_policy="mp20-standard",
            )
            self.assertEqual(manifest["split_policy"], "mp20-standard")
            self.assertEqual(manifest["chemsys_overlap"], {"train__val": 1})
            self.assertFalse(manifest["chemsys_held_out"])
            self.assertTrue(manifest["gate"]["split_policy_honored"])
            self.assertTrue(manifest["gate"]["chemsys_overlap_disclosed"])
            self.assertTrue(all(manifest["gate"].values()))

    def test_original_source_file_ordinal_is_audited_when_index_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            for split in ("train", "val"):
                rows = read_jsonl(source / f"{split}.jsonl")
                rows[0].pop("source_row_idx")
                rows[0].pop("c3fd_certificate_source_row_idx")
                write_jsonl(source / f"{split}.jsonl", rows)

            output = root / "ordinal-source"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
                allow_source_ordinal_index=True,
            )
            self.assertEqual(
                manifest["source_index_policy"], "file_ordinal_if_absent"
            )
            for split in ("train", "val"):
                self.assertEqual(
                    manifest["splits"][split][
                        "source_rows_assigned_ordinal_index"
                    ],
                    1,
                )
                self.assertEqual(
                    {row["source_row_idx"] for row in read_jsonl(output / f"{split}.jsonl")},
                    {0},
                )

    def test_unsupervised_mp20_row_is_not_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            rows = read_jsonl(semantic / "train.jsonl")
            rows[0]["composition_supervision"] = False
            write_jsonl(semantic / "train.jsonl", rows)

            output = root / "unsupervised-row-retained"
            manifest = MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=output,
            )
            teacher = next(
                row
                for row in read_jsonl(output / "train.jsonl")
                if row["view"] == "teacher-native"
            )
            parsed = MODULE.parse_native_plan_line(teacher["native_plan_line"])
            self.assertEqual(parsed["elements"], ["Li", "O"])
            self.assertEqual(parsed["counts"], [2, 1])
            self.assertNotIn("CB=", teacher["native_plan_line"])
            self.assertNotIn(":Q", teacher["native_plan_line"])
            self.assertNotIn("composition_supervision_authorized", teacher)
            self.assertEqual(manifest["splits"]["train"]["source_rows"], 1)
            self.assertTrue(all(manifest["gate"].values()))

    def test_charge_and_valence_auxiliary_fields_do_not_change_native_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, semantic, predicted = self.make_formal_inputs(root)
            clean_output = root / "clean-native"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=clean_output,
            )
            clean_teacher = next(
                row
                for row in read_jsonl(clean_output / "train.jsonl")
                if row["view"] == "teacher-native"
            )

            source_rows = read_jsonl(source / "train.jsonl")
            source_rows[0]["plan_state"]["charge_bucket"] = "charge_fail"
            write_jsonl(source / "train.jsonl", source_rows)
            semantic_rows = read_jsonl(semantic / "train.jsonl")
            semantic_rows[0]["plan_state"]["charge_bucket"] = "pauling_fail"
            semantic_rows[0]["plan_state"]["valence_species"] = [
                {"element": "Li", "count": 1, "oxidation_state": 7},
            ]
            write_jsonl(semantic / "train.jsonl", semantic_rows)
            mutated_output = root / "mutated-auxiliary"
            MODULE.build_dataset(
                input_dir=source,
                semantic_dir=semantic,
                predicted_soft_dir=predicted,
                output_dir=mutated_output,
            )
            mutated_teacher = next(
                row
                for row in read_jsonl(mutated_output / "train.jsonl")
                if row["view"] == "teacher-native"
            )
            self.assertEqual(
                clean_teacher["native_plan_line"], mutated_teacher["native_plan_line"]
            )
            self.assertNotIn("CB=", mutated_teacher["native_plan_line"])
            self.assertNotIn(":Q", mutated_teacher["native_plan_line"])


if __name__ == "__main__":
    unittest.main()

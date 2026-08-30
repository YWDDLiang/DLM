import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_c3fd_native_soft_predictions",
    ROOT / "scripts" / "export_c3fd_native_soft_predictions.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import export_c3fd_native_soft_predictions.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ExportC3FDNativeSoftPredictionsTest(unittest.TestCase):
    def vocabulary(self):
        return {
            "schema": MODULE.SEMANTIC_VOCABULARY_SCHEMA,
            "species": [
                {"id": 0, "atomic_number": 3, "oxidation_state": 1},
                {"id": 1, "atomic_number": 8, "oxidation_state": -2},
                {"id": 2, "atomic_number": 11, "oxidation_state": 1},
                {"id": 3, "atomic_number": 17, "oxidation_state": -1},
            ],
            "species_eos_id": 4,
            "physics": {"matrix": [[0.0], [1.0], [2.0], [3.0]]},
            "soft_vocabulary": {
                "anion_framework": ["oxide", "halide", MODULE.UNKNOWN_LABEL],
                "charge_bucket": ["neutral_plausible", MODULE.UNKNOWN_LABEL],
                "lattice_system": ["cubic", "hexagonal", MODULE.UNKNOWN_LABEL],
                "spacegroup_bucket": [
                    "sg_003_015",
                    "sg_195_230",
                    MODULE.UNKNOWN_LABEL,
                ],
                "volume_per_atom_bin": [
                    "volpa_015_019",
                    "volpa_020_024",
                    MODULE.UNKNOWN_LABEL,
                ],
            },
        }

    def semantic_row(self, split, source_row_idx, variant):
        if variant == 0:
            plan = {
                "N": 3,
                "elements": ["Li", "O"],
                "counts": [2, 1],
                "metadata": {"e_above_hull": 0.0},
                "material_id": "must-not-leak-mp-1",
                "cif": "must-not-leak-cif",
            }
            soft_labels = {
                "anion_framework": 0,
                "charge_bucket": 0,
                "lattice_system": 0,
                "spacegroup_bucket": 1,
                "volume_per_atom_bin": 0,
            }
        else:
            plan = {
                "N": 2,
                "elements": ["Na", "Cl"],
                "counts": [1, 1],
                "target_stability": "must-not-leak-stability",
                "energy": -1.0,
                "generated_outcome": "must-not-leak-outcome",
            }
            soft_labels = {
                "anion_framework": 1,
                "charge_bucket": 0,
                "lattice_system": 1,
                "spacegroup_bucket": 0,
                "volume_per_atom_bin": 1,
            }
        return {
            "schema": MODULE.SEMANTIC_ROW_SCHEMA,
            "split": split,
            "source_row_idx": source_row_idx,
            "proposal_supervision": True,
            "composition_supervision": True,
            "plan_state": plan,
            "proposal_targets": {
                "family": soft_labels["anion_framework"],
                "N": plan["N"],
                "arity": len(plan["elements"]),
            },
            "species_labels": [0, 1],
            "count_targets": plan["counts"],
            "ledger_steps": [],
            "soft_labels": soft_labels,
            "body": "must-not-leak-body",
            "hull": "must-not-leak-hull",
        }

    def make_semantic_data(self, root):
        data_dir = root / "semantic"
        data_dir.mkdir()
        vocabulary = self.vocabulary()
        write_json(data_dir / "vocabulary.json", vocabulary)
        for split, first_index in (("train", 10), ("val", 20)):
            write_jsonl(
                data_dir / f"{split}.jsonl",
                [
                    self.semantic_row(split, first_index, 0),
                    self.semantic_row(split, first_index + 1, 1),
                ],
            )
        manifest = {
            "schema": MODULE.SEMANTIC_MANIFEST_SCHEMA,
            "vocabulary": vocabulary,
            "splits": {"train": {"rows": 2}, "val": {"rows": 2}},
            "gate": {"planner_training_data_authorized": True},
        }
        write_json(data_dir / "manifest.json", manifest)
        return data_dir

    def prediction(self, value, confidence):
        return {"prediction": value, "confidence": confidence}

    def checkpoint_result(self, root, semantic_inputs, name, seed, offset=0):
        path = root / f"{name}.pt"
        path.write_bytes(f"frozen checkpoint {name}".encode("utf-8"))
        labels = semantic_inputs["labels"]
        split_results = {}
        for split in MODULE.REQUIRED_SPLITS:
            indices = [
                row["source_row_idx"] for row in semantic_inputs["splits"][split]["rows"]
            ]
            lattice = [
                self.prediction(labels["lattice_system"][(offset + i) % 2], 0.90 - i * 0.1)
                for i in range(2)
            ]
            # Deliberately independent of lattice: cubic may pair with monoclinic SG.
            spacegroup = [
                self.prediction(labels["spacegroup_bucket"][(offset + i + 1) % 2], 0.80)
                for i in range(2)
            ]
            volume = [
                self.prediction(labels["volume_per_atom_bin"][(offset + i) % 2], 0.70)
                for i in range(2)
            ]
            if name == "seed18" and split == "val":
                volume[1] = self.prediction(MODULE.UNKNOWN_LABEL, 0.51)
            split_results[split] = {
                "source_row_indices": indices,
                "predictions": {
                    "lattice_system": lattice,
                    "spacegroup_bucket": spacegroup,
                    "volume_per_atom_bin": volume,
                },
            }
        return {
            "name": name,
            "seed": seed,
            "checkpoint_schema": MODULE.CHECKPOINT_SCHEMA,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config_sha256": "a" * 64,
            "vocabulary_sha256": semantic_inputs["vocabulary_sha256"],
            "device": "cpu",
            "confidence_definition": "maximum_uncalibrated_softmax_probability",
            "splits": split_results,
        }

    def fixture(self, root):
        data_dir = self.make_semantic_data(root)
        semantic_inputs = MODULE.load_semantic_inputs(data_dir)
        results = [
            self.checkpoint_result(root, semantic_inputs, "seed17", 17, offset=0),
            self.checkpoint_result(root, semantic_inputs, "seed18", 18, offset=1),
        ]
        return semantic_inputs, results

    def test_preserves_both_checkpoints_without_selection_and_reports_unknowns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            output = root / "output"
            manifest = MODULE.write_export(
                semantic_inputs=semantic_inputs,
                checkpoint_results=results,
                output_dir=output,
            )
            rows = read_jsonl(output / "train.jsonl")
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(
                    set(row["predictions_by_checkpoint"]), {"seed17", "seed18"}
                )
                self.assertEqual(
                    set(row["predictions_by_checkpoint"]["seed17"]),
                    set(MODULE.PREDICTION_FIELDS),
                )
            self.assertEqual(manifest["selection"], "none")
            self.assertTrue(manifest["all_frozen_checkpoints_preserved"])
            self.assertTrue(manifest["independent_spacegroup_head"])
            self.assertEqual(
                manifest["splits"]["val"]["predictions"]["seed18"]
                ["volume_per_atom_bin"]["unknown_predictions"],
                1,
            )
            self.assertEqual(
                manifest["splits"]["train"]["predictions"]["seed17"]
                ["lattice_system"]["class_distribution"]["cubic"],
                1,
            )
            for filename in ("train.jsonl", "val.jsonl", "manifest.json"):
                self.assertTrue((output / filename).is_file())
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, filename = line.split("  ", 1)
                self.assertEqual(MODULE.sha256_file(output / filename), digest)
            self.assertTrue((output / "_SUCCESS").is_file())

    def test_curated_rows_copy_no_metadata_ids_energy_hull_stability_or_structures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            output = root / "output"
            MODULE.write_export(
                semantic_inputs=semantic_inputs,
                checkpoint_results=results,
                output_dir=output,
            )
            for split in MODULE.REQUIRED_SPLITS:
                for row in read_jsonl(output / f"{split}.jsonl"):
                    self.assertEqual(
                        set(row),
                        {
                            "schema",
                            "split",
                            "source_row_idx",
                            "N",
                            "elements",
                            "counts",
                            "predictions_by_checkpoint",
                        },
                    )
                    serialised = json.dumps(row, sort_keys=True).lower()
                    for forbidden in (
                        "metadata",
                        "material_id",
                        "must-not-leak",
                        "energy",
                        "hull",
                        "stability",
                        "cif",
                        "body",
                        "outcome",
                    ):
                        self.assertNotIn(forbidden, serialised)

    def test_checkpoint_row_order_mismatch_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            broken = copy.deepcopy(results)
            broken[1]["splits"]["val"]["source_row_indices"].reverse()
            output = root / "misaligned"
            with self.assertRaisesRegex(ValueError, "row/order mismatch"):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=broken,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_field_and_vocabulary_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            missing_field = copy.deepcopy(results)
            del missing_field[0]["splits"]["train"]["predictions"]["spacegroup_bucket"]
            with self.assertRaisesRegex(ValueError, "prediction fields mismatch"):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=missing_field,
                    output_dir=root / "missing-field",
                )
            self.assertFalse((root / "missing-field").exists())

            bad_vocab = copy.deepcopy(results)
            bad_vocab[1]["splits"]["train"]["predictions"]["lattice_system"][0] = (
                self.prediction("not-in-vocabulary", 0.9)
            )
            with self.assertRaisesRegex(ValueError, "outside vocabulary"):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=bad_vocab,
                    output_dir=root / "bad-vocab",
                )
            self.assertFalse((root / "bad-vocab").exists())

    def test_checkpoint_seed_or_config_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            bad_seed = copy.deepcopy(results)
            bad_seed[1]["seed"] = 17
            with self.assertRaisesRegex(ValueError, "exactly 17 and 18"):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=bad_seed,
                    output_dir=root / "bad-seed",
                )
            bad_config = copy.deepcopy(results)
            bad_config[1]["config_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "configs mismatch"):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=bad_config,
                    output_dir=root / "bad-config",
                )

    def test_semantic_source_order_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = self.make_semantic_data(root)
            rows = read_jsonl(data_dir / "val.jsonl")
            rows.reverse()
            write_jsonl(data_dir / "val.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "source_row_idx order mismatch"):
                MODULE.load_semantic_inputs(data_dir)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            semantic_inputs, results = self.fixture(root)
            output = root / "output"
            MODULE.write_export(
                semantic_inputs=semantic_inputs,
                checkpoint_results=results,
                output_dir=output,
            )
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                MODULE.write_export(
                    semantic_inputs=semantic_inputs,
                    checkpoint_results=results,
                    output_dir=output,
                )
            self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

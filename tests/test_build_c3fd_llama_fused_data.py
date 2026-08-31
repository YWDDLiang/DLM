from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_c3fd_llama_fused_data",
    ROOT / "scripts" / "build_c3fd_llama_fused_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import fused data builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

from crystal_dlm.ccfd import FormulaToken
import crystal_dlm.c3fd_llama_fused_plan as FUSED_MODULE


class FixtureReachability:
    def __init__(self, _nodes):
        pass

    def legal_species_counts(self, state, **_kwargs):
        if len(state.tokens) == 0:
            return (FormulaToken(11, 1, 1),)
        if len(state.tokens) == 1:
            return (FormulaToken(17, -1, 1),)
        return ()


FUSED_MODULE.PaulingBitsetReachability = FixtureReachability


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def vocabulary():
    return {
        "count_values": list(range(1, 21)),
        "species": [
            {"id": 0, "atomic_number": 11, "oxidation_state": 1},
            {"id": 1, "atomic_number": 17, "oxidation_state": -1},
        ],
        "soft_vocabulary": {
            "anion_framework": ["oxide", "halide"],
            "lattice_system": ["cubic"],
            "spacegroup_bucket": ["sg_195_230"],
            "volume_per_atom_bin": ["volpa_020_024"],
        },
    }


def semantic_row(index: int, *, split: str | None = None):
    row = {
        "source_row_idx": index,
        "sample_weight": 1.0,
        "certificate_class": "benchmark_compatible",
        "composition_supervision": True,
        "proposal_supervision": True,
        "compile_error": None,
        "N_target": 2,
        "proposal_targets": {"family": 1, "N": 2, "arity": 2},
        "species_labels": [0, 1],
        "count_targets": [1, 1],
        "ledger_steps": [
            {
                "remaining_atoms": 2,
                "net_charge": 0,
                "remaining_species": 2,
                "branch": "unset",
            },
            {
                "remaining_atoms": 2,
                "net_charge": 0,
                "remaining_species": 2,
                "branch": "unset",
            },
            {
                "remaining_atoms": 1,
                "net_charge": 1,
                "remaining_species": 1,
                "branch": "ionic",
            },
            {
                "remaining_atoms": 0,
                "net_charge": 0,
                "remaining_species": 0,
                "branch": "ionic",
            },
        ],
        "soft_labels": {
            "anion_framework": 1,
            "lattice_system": 0,
            "spacegroup_bucket": 0,
            "volume_per_atom_bin": 0,
        },
        "plan_state": {
            "N": 2,
            "elements": ["Na", "Cl"],
            "counts": [1, 1],
            "anion_framework": "halide",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_020_024",
            "cif": "forbidden",
            "e_above_hull": -8.0,
        },
        "answer": "forbidden crystal body",
    }
    if split is not None:
        row["source_split"] = split
    return row


def source_row(index: int, hull, *, split: str | None = None):
    row = {
        "c3fd_certificate_source_row_idx": index,
        "metadata": {
            "e_above_hull": hull,
            "official_e_above_hull": -123,
            "material_id": "must-not-copy",
        },
        "minimal_spec": {"N": 999},
        "prompt": "forbidden source prompt",
        "answer": "forbidden dynamic crystal body",
        "structure": {"lattice": "forbidden"},
        "cif": "forbidden",
    }
    if split is not None:
        row["source_split"] = split
    return row


class Workspace:
    def __init__(self, root: Path):
        self.semantic = root / "semantic"
        self.ctv = root / "ctv"
        self.output = root / "output"
        self.semantic.mkdir()
        self.ctv.mkdir()
        (self.semantic / "vocabulary.json").write_text(
            json.dumps(vocabulary()), encoding="utf-8"
        )

    def write_split(self, split: str, semantic_rows, source_rows):
        write_jsonl(self.semantic / f"{split}.jsonl", semantic_rows)
        write_jsonl(self.ctv / f"{split}.jsonl", source_rows)

    def make_valid(self):
        self.write_split("train", [semantic_row(1)], [source_row(1, 0.0)])
        self.write_split("val", [semantic_row(1)], [source_row(1, 0.2)])


class BuildFusedDataTest(unittest.TestCase):
    def test_keyed_join_not_ordinal_zip_and_split_local_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            # Deliberately reverse the source order. Values must follow keys.
            workspace.write_split(
                "train",
                [semantic_row(10), semantic_row(20)],
                [source_row(20, 0.2), source_row(10, 0.1)],
            )
            # The same keys in val carry different split-local metadata.
            workspace.write_split(
                "val",
                [semantic_row(20), semantic_row(10)],
                [source_row(10, 0.100001), source_row(20, -0.1)],
            )
            manifest = MODULE.build_dataset(
                semantic_dir=workspace.semantic,
                ctv_minimal_dir=workspace.ctv,
                output_dir=workspace.output,
            )
            train = read_jsonl(workspace.output / "train.jsonl")
            val = read_jsonl(workspace.output / "val.jsonl")
            self.assertEqual([row["source_row_idx"] for row in train], [10, 20])
            self.assertEqual(
                [row["stability_condition"] for row in train],
                ["meta_or_better", "higher"],
            )
            self.assertEqual(
                [row["stability_condition"] for row in val],
                ["higher", "meta_or_better"],
            )
            self.assertEqual(
                manifest["join_key"],
                "semantic.source_row_idx == ctv_minimal.c3fd_certificate_source_row_idx",
            )
            self.assertFalse(manifest["ordinal_zip_used"])

    def test_primary_typed_fields_and_no_source_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.make_valid()
            MODULE.build_dataset(
                semantic_dir=workspace.semantic,
                ctv_minimal_dir=workspace.ctv,
                output_dir=workspace.output,
            )
            row = read_jsonl(workspace.output / "train.jsonl")[0]
            self.assertEqual(
                set(row),
                {
                    "schema",
                    "target_schema",
                    "source_split",
                    "source_row_idx",
                    "sample_weight",
                    "stability_condition",
                    "proposal_target",
                    "species_ids",
                    "count_targets",
                    "ledger_steps",
                    "soft_targets",
                    "audit_transcript",
                },
            )
            serialized = json.dumps(row, sort_keys=True)
            for forbidden in (
                "e_above_hull",
                "material_id",
                "dynamic crystal body",
                "forbidden source prompt",
                '"cif"',
                '"structure"',
                '"plan_state"',
                '"metadata"',
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(row["species_ids"], [0, 1])
            self.assertEqual(row["count_targets"], [1, 1])
            self.assertEqual(row["soft_targets"]["volume_per_atom_bin"]["label"], 0)

    def test_all_skips_are_accounted_without_silent_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            invalid_action = semantic_row(3)
            invalid_action["composition_supervision"] = False
            missing_metadata = source_row(2, 0.0)
            missing_metadata.pop("metadata")
            workspace.write_split(
                "train",
                [semantic_row(1), semantic_row(2), invalid_action, semantic_row(5)],
                [source_row(1, 0.0), missing_metadata, source_row(3, 0.0), source_row(4, 0.0)],
            )
            workspace.write_split("val", [semantic_row(1)], [source_row(1, 0.0)])
            manifest = MODULE.build_dataset(
                semantic_dir=workspace.semantic,
                ctv_minimal_dir=workspace.ctv,
                output_dir=workspace.output,
            )
            report = manifest["splits"]["train"]
            self.assertEqual(report["join_union_rows"], 5)
            self.assertEqual(report["kept_rows"], 1)
            self.assertEqual(report["skipped_rows"], 4)
            self.assertEqual(
                report["skipped_reasons"],
                {
                    "invalid_teacher_action_sequence": 1,
                    "missing_ctv_minimal_row": 1,
                    "missing_semantic_row": 1,
                    "missing_source_metadata": 1,
                },
            )
            self.assertEqual(
                report["skipped_source_row_indices"]["missing_semantic_row"], [4]
            )
            self.assertEqual(
                report["skipped_source_row_indices"]["missing_ctv_minimal_row"], [5]
            )

    def test_malformed_missing_and_nonfinite_metadata_have_distinct_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_path = root / "semantic.jsonl"
            source_path = root / "source.jsonl"
            write_jsonl(semantic_path, [semantic_row(index) for index in range(1, 6)])
            no_key = source_row(2, 0.0)
            no_key["metadata"].pop("e_above_hull")
            no_metadata = source_row(3, 0.0)
            no_metadata.pop("metadata")
            write_jsonl(
                source_path,
                [
                    source_row(1, 0.0),
                    no_key,
                    no_metadata,
                    source_row(4, "bad"),
                    source_row(5, math.nan),
                ],
            )
            rows, report = MODULE.build_split_rows(
                split="train",
                semantic_path=semantic_path,
                source_path=source_path,
                vocabulary=vocabulary(),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                report["skipped_reasons"],
                {
                    "malformed_e_above_hull": 1,
                    "missing_e_above_hull": 1,
                    "missing_source_metadata": 1,
                    "nonfinite_e_above_hull": 1,
                },
            )

    def test_split_marker_mismatch_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_path = root / "semantic.jsonl"
            source_path = root / "source.jsonl"
            write_jsonl(semantic_path, [semantic_row(1), semantic_row(2, split="val")])
            write_jsonl(source_path, [source_row(1, 0.0), source_row(2, 0.0)])
            rows, report = MODULE.build_split_rows(
                split="train",
                semantic_path=semantic_path,
                source_path=source_path,
                vocabulary=vocabulary(),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(report["skipped_reasons"], {"split_marker_mismatch": 1})

    def test_duplicate_or_missing_join_keys_fail_hard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [semantic_row(1), semantic_row(1)])
            with self.assertRaisesRegex(ValueError, "duplicates source_row_idx"):
                MODULE.index_rows(path, label="semantic:train")
            row = semantic_row(2)
            row.pop("source_row_idx")
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "lacks immutable source_row_idx"):
                MODULE.index_rows(path, label="semantic:train")

    def test_output_is_immutable_hashed_and_train_val_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.make_valid()
            # These files must never be inspected or represented in the manifest.
            (workspace.semantic / "test.jsonl").write_text("not json\n", encoding="utf-8")
            (workspace.ctv / "prospective.jsonl").write_text("not json\n", encoding="utf-8")
            manifest = MODULE.build_dataset(
                semantic_dir=workspace.semantic,
                ctv_minimal_dir=workspace.ctv,
                output_dir=workspace.output,
            )
            self.assertEqual(manifest["included_splits"], ["train", "val"])
            self.assertEqual(manifest["dev_test_prospective_rows"], 0)
            self.assertEqual(
                set(manifest["input_sha256"]),
                {
                    "semantic_vocabulary",
                    "semantic_train",
                    "semantic_val",
                    "ctv_minimal_train",
                    "ctv_minimal_val",
                },
            )
            self.assertEqual(
                manifest["splits"]["train"]["stability_condition_counts"],
                {"meta_or_better": 1},
            )
            self.assertEqual(
                manifest["splits"]["val"]["stability_condition_counts"],
                {"higher": 1},
            )
            self.assertTrue((workspace.output / "_SUCCESS").is_file())
            sums = (workspace.output / "SHA256SUMS").read_text(encoding="utf-8")
            for line in sums.splitlines():
                expected, name = line.split("  ", 1)
                actual = hashlib.sha256((workspace.output / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            success = json.loads((workspace.output / "_SUCCESS").read_text())
            self.assertEqual(
                success["manifest_sha256"],
                hashlib.sha256((workspace.output / "manifest.json").read_bytes()).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                MODULE.build_dataset(
                    semantic_dir=workspace.semantic,
                    ctv_minimal_dir=workspace.ctv,
                    output_dir=workspace.output,
                )

    def test_failed_build_preserves_negative_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.write_split("train", [], [])
            workspace.write_split("val", [semantic_row(1)], [source_row(1, 0.0)])
            with self.assertRaisesRegex(ValueError, "train has no valid"):
                MODULE.build_dataset(
                    semantic_dir=workspace.semantic,
                    ctv_minimal_dir=workspace.ctv,
                    output_dir=workspace.output,
                )
            failed = workspace.output.with_name(f".{workspace.output.name}.FAILED")
            self.assertTrue(failed.is_dir())
            self.assertFalse(workspace.output.exists())


if __name__ == "__main__":
    unittest.main()

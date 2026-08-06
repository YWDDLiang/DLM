from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.dataset import (
    PymatgenWyckoffCodec,
    audit_wq_dataset,
    build_hash_fixed_subset,
    material_family_from_symbols,
    tolerance_tag,
)


PYXTAL_AVAILABLE = importlib.util.find_spec("pyxtal") is not None


def _record(split: str, material_id: str, canonical_hash: str) -> dict:
    primary = {
        "roundtrip_match": True,
        "multiplicity_consistent": True,
        "canonical_hash": canonical_hash,
        "orbits": [
            {
                "orbit": {"chart_dimension": 0},
                "chart_basis": [[], [], []],
                "chart_fit_residual_angstrom": 0.0,
                "primitive_chart_jacobians": [[[], [], []]],
            }
        ],
        "state": {
            "orbits": [
                {
                    "multiplicity": 1,
                    "primitive_multiplicity": 1,
                }
            ]
        },
    }
    return {
        "schema": "mp20_wq_v1",
        "material_id": material_id,
        "split": split,
        "source_elements": ["Fe", "O"],
        "material_family": "oxide",
        "selected": True,
        "ambiguous": False,
        "primary_failure_reason": "",
        "decompositions": {tolerance_tag(1.0e-2): primary},
    }


class DatasetAuditTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def test_small_registered_dataset_passes_when_count_gate_is_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            self._write(train, [_record("train", "m1", "h1")])
            self._write(val, [_record("val", "m2", "h2")])
            result = audit_wq_dataset(
                {"train": [train], "val": [val]},
                allow_nonpaper_counts=True,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["canonical_overlap_count"], 0)
            self.assertEqual(result["splits"]["train"]["coverage"], 1.0)

    def test_cross_split_canonical_leakage_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            self._write(train, [_record("train", "m1", "shared")])
            self._write(test, [_record("test", "m2", "shared")])
            result = audit_wq_dataset(
                {"train": [train], "test": [test]},
                allow_nonpaper_counts=True,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["canonical_overlap_count"], 1)

    def test_material_family_names_match_the_protocol(self) -> None:
        self.assertEqual(material_family_from_symbols(["Li", "O"]), "oxide")
        self.assertEqual(
            material_family_from_symbols(["Li", "O", "F"]), "mixed_anion"
        )

    def test_hash_fixed_subset_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "first.jsonl"
            second_source = root / "second.jsonl"
            rows = [_record("train", f"m{index}", f"h{index}") for index in range(10)]
            self._write(first_source, rows)
            self._write(second_source, list(reversed(rows)))
            first = build_hash_fixed_subset(
                [first_source], output_path=root / "subset1.jsonl", count=4
            )
            second = build_hash_fixed_subset(
                [second_source], output_path=root / "subset2.jsonl", count=4
            )
            self.assertEqual(
                first["selected_material_id_hash"], second["selected_material_id_hash"]
            )
            self.assertEqual(first["output_sha256"], second["output_sha256"])

    @unittest.skipUnless(
        PYXTAL_AVAILABLE,
        "PyXtal 1.1.4 setting alignment is verified in the locked server env",
    )
    def test_pnma_ita_setting_does_not_apply_band_structure_axis_sorting(self) -> None:
        from pymatgen.core import Lattice, Structure

        structure = Structure.from_spacegroup(
            "Pnma",
            Lattice.orthorhombic(7.1, 5.2, 6.3),
            ["Na", "Cl"],
            [[0.17, 0.25, 0.31], [0.0, 0.0, 0.0]],
        )
        record = PymatgenWyckoffCodec().from_cif(
            cif=structure.to(fmt="cif"),
            material_id="synthetic-pnma",
            split="val",
        )
        self.assertTrue(record.selected, record.primary_failure_reason)
        self.assertEqual(record.primary.state.space_group, 62)
        self.assertLess(
            max(orbit.chart_fit_residual_angstrom for orbit in record.primary.orbits),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()

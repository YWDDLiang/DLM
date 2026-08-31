from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from crystal_dlm.c3fd_llama_fused_plan import (
    STABILITY_HIGHER,
    STABILITY_META_OR_BETTER,
    audit_transcript_from_targets,
    stability_condition_from_e_above_hull,
    typed_targets_from_semantic_row,
)


def vocabulary():
    return {
        "species": [
            {"id": 4, "atomic_number": 11, "oxidation_state": 1},
            {"id": 9, "atomic_number": 17, "oxidation_state": -1},
        ],
        "soft_vocabulary": {
            "anion_framework": ["oxide", "halide"],
            "lattice_system": ["cubic", "orthorhombic"],
            "spacegroup_bucket": ["sg_195_230", "sg_016_074"],
            "volume_per_atom_bin": ["volpa_020_024", "volpa_025_029"],
        },
    }


def semantic_row():
    return {
        "source_row_idx": 12,
        "sample_weight": 1.0,
        "certificate_class": "benchmark_compatible",
        "composition_supervision": True,
        "proposal_supervision": True,
        "compile_error": None,
        "N_target": 2,
        "proposal_targets": {"family": 1, "N": 2, "arity": 2},
        "species_labels": [4, 9],
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
            "structure": {"must": "not leak"},
            "e_above_hull": -99,
        },
    }


class StabilityConditionTest(unittest.TestCase):
    def test_binary_tier_boundary_is_inclusive(self):
        self.assertEqual(
            stability_condition_from_e_above_hull(-0.2),
            STABILITY_META_OR_BETTER,
        )
        self.assertEqual(
            stability_condition_from_e_above_hull(0.1),
            STABILITY_META_OR_BETTER,
        )
        self.assertEqual(
            stability_condition_from_e_above_hull(0.1000001), STABILITY_HIGHER
        )

    def test_missing_malformed_and_nonfinite_are_rejected(self):
        for value, message in (
            (None, "missing"),
            ("", "missing"),
            (True, "malformed"),
            ("not-a-number", "malformed"),
            (math.nan, "nonfinite"),
            (math.inf, "nonfinite"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                stability_condition_from_e_above_hull(value)


class TypedTargetsTest(unittest.TestCase):
    def test_extracts_typed_residual_head_targets(self):
        targets = typed_targets_from_semantic_row(semantic_row(), vocabulary())
        self.assertEqual(
            targets["proposal_target"],
            {"family_id": 1, "family_value": "halide", "N": 2, "arity": 2},
        )
        self.assertEqual(targets["species_ids"], [4, 9])
        self.assertEqual(targets["count_targets"], [1, 1])
        self.assertEqual(
            targets["species_actions"],
            [
                {"atomic_number": 11, "oxidation_state": 1},
                {"atomic_number": 17, "oxidation_state": -1},
            ],
        )
        self.assertEqual(
            targets["soft_targets"]["lattice_system"],
            {"label": 0, "value": "cubic"},
        )
        self.assertNotIn("plan_state", targets)
        self.assertNotIn("e_above_hull", str(targets))

    def test_audit_transcript_is_deterministic_and_complete(self):
        first = typed_targets_from_semantic_row(semantic_row(), vocabulary())
        second = typed_targets_from_semantic_row(semantic_row(), vocabulary())
        expected = (
            "PROPOSAL family=halide family_id=1 N=2 arity=2\n"
            "SPECIES id=4 Z=11 oxidation=+1 count=1\n"
            "SPECIES id=9 Z=17 oxidation=-1 count=1\n"
            "EOS_COMPOSITION\n"
            "SOFT field=lattice_system label=0 value=cubic\n"
            "SOFT field=spacegroup_bucket label=0 value=sg_195_230\n"
            "SOFT field=volume_per_atom_bin label=0 value=volpa_020_024\n"
            "END_TYPED_PLAN\n"
        )
        self.assertEqual(first["audit_transcript"], expected)
        self.assertEqual(first["audit_transcript"], second["audit_transcript"])
        self.assertEqual(audit_transcript_from_targets(first), expected)

    def test_invalid_teacher_sequences_fail_closed(self):
        mutations = []
        invalid_certificate = semantic_row()
        invalid_certificate["composition_supervision"] = False
        mutations.append(invalid_certificate)
        bad_count = semantic_row()
        bad_count["count_targets"] = [2, 1]
        mutations.append(bad_count)
        bad_order = semantic_row()
        bad_order["species_labels"] = [9, 4]
        mutations.append(bad_order)
        bad_ledger = semantic_row()
        bad_ledger["ledger_steps"][-1]["net_charge"] = 2
        mutations.append(bad_ledger)
        for row in mutations:
            with self.subTest(row=row), self.assertRaises(ValueError):
                typed_targets_from_semantic_row(row, vocabulary())

    def test_soft_label_value_mismatch_and_control_text_are_rejected(self):
        mismatch = semantic_row()
        mismatch["soft_labels"]["lattice_system"] = 1
        with self.assertRaisesRegex(ValueError, "label/value mismatch"):
            typed_targets_from_semantic_row(mismatch, vocabulary())
        injected = copy.deepcopy(semantic_row())
        injected["plan_state"]["lattice_system"] = "cubic\nBODY"
        with self.assertRaisesRegex(ValueError, "one non-empty token"):
            typed_targets_from_semantic_row(injected, vocabulary())


if __name__ == "__main__":
    unittest.main()

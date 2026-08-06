from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


THIS_DIR = Path(__file__).resolve().parent
INNOVATION_ROOT = THIS_DIR.parent
CODE_ROOT = INNOVATION_ROOT / "code"
REACTIVATION_ROOT = INNOVATION_ROOT.parent
PROJECT_ROOT = INNOVATION_ROOT.parents[2]
RESTORED_BASELINE_ROOT = REACTIVATION_ROOT / "baseline"
RUNTIME_ROOT = RESTORED_BASELINE_ROOT if (RESTORED_BASELINE_ROOT / "crystal_dlm").is_dir() else PROJECT_ROOT
for value in (str(CODE_ROOT), str(RUNTIME_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from jointchem_data import (  # noqa: E402
    _proportional_stratified_selection,
    assert_no_leakage,
    attach_joint_negatives,
    build_training_stream,
    build_validation_panel,
    donor_bucket,
    enrich_rows,
    geometry_tuple,
    make_chemistry_negative,
    make_joint_negative,
    public_row,
    write_jsonl,
)
from jointchem_loss import (  # noqa: E402
    build_target_only_token_lists,
    supervised_target_positions,
)


def synthetic_classifier(elems, counts):
    key = (tuple(int(v) for v in elems), tuple(int(v) for v in counts))
    valid = {
        ((3, 8), (2, 1)),
        ((11, 17), (1, 1)),
        ((12, 8), (1, 1)),
        ((13, 8), (2, 3)),
    }
    if key in valid:
        return {"valid": True, "reason": "charge_neutral_pauling_valid"}
    return {"valid": False, "reason": "charge_neutrality_fail"}


def plan(
    formula,
    elements,
    counts,
    *,
    lattice="cubic",
    spacegroup="sg_195_230",
    volume="volpa_010_014",
    anion="oxide",
):
    return {
        "formula": formula,
        "reduced_formula": formula,
        "elements": list(elements),
        "counts": list(counts),
        "N": sum(counts),
        "anion_framework": anion,
        "charge_bucket": "neutral_plausible",
        "lattice_system": lattice,
        "spacegroup_bucket": spacegroup,
        "volume_per_atom_bin": volume,
    }


def answer(value):
    return "\n".join(
        [
            f"formula: {value['formula']}",
            f"anion: {value['anion_framework']}",
            f"charge: {value['charge_bucket']}",
            f"lattice: {value['lattice_system']}",
            f"spacegroup: {value['spacegroup_bucket']}",
            f"volume: {value['volume_per_atom_bin']}",
            "end: plan",
        ]
    )


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [1 + (ord(char) % 89) for char in str(text)]}


class JointChemDataTests(unittest.TestCase):
    def test_chemistry_negative_preserves_N_arity_elements_and_rich_fields(self):
        source = plan("Li2O", ["Li", "O"], [2, 1])
        text, audit = make_chemistry_negative(
            source,
            row_key="train:0:abc",
            seed=17,
            classifier=synthetic_classifier,
        )
        self.assertTrue(audit["available"])
        self.assertEqual(audit["N"], 3)
        self.assertEqual(audit["arity"], 2)
        self.assertIn("formula: LiO2", text)
        self.assertIn("anion: oxide", text)
        self.assertIn("charge: neutral_plausible", text)
        self.assertIn("lattice: cubic", text)

    def test_joint_negative_changes_only_geometry_tuple(self):
        source = plan("Li2O", ["Li", "O"], [2, 1])
        donor = plan(
            "MgO",
            ["Mg", "O"],
            [1, 1],
            lattice="hexagonal",
            spacegroup="sg_168_194",
            volume="volpa_005_009",
        )
        self.assertEqual(donor_bucket(source), donor_bucket(donor))
        text, audit = make_joint_negative(source, donor)
        self.assertTrue(audit["available"])
        self.assertIn("formula: Li2O", text)
        self.assertIn("anion: oxide", text)
        self.assertIn("charge: neutral_plausible", text)
        self.assertIn("lattice: hexagonal", text)
        self.assertNotEqual(geometry_tuple(source), tuple(audit["negative_geometry"]))

    def test_stream_is_exact_and_deterministic(self):
        rows = []
        for index in range(20):
            rows.append(
                {
                    "row_key": f"train:{index}",
                    "composition_valid": True,
                    "composition_reason": "charge_neutral_pauling_valid",
                    "chemistry_negative_answer": "negative",
                    "joint_negative_answer": "joint",
                }
            )
        first = build_training_stream(rows, total_rows=10, positive_fraction=0.8, seed=17)
        second = build_training_stream(rows, total_rows=10, positive_fraction=0.8, seed=17)
        self.assertEqual(first, second)
        roles = [row["stream_role"] for row in first]
        self.assertEqual(roles.count("chemistry_valid_positive"), 8)
        self.assertEqual(roles.count("epoch2_anchor"), 2)

    def test_valid_selection_does_not_inflate_all_metal_shortcut(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "row_key": f"train:neutral:{index}",
                    "composition_valid": True,
                    "composition_reason": "charge_neutral_pauling_valid",
                }
            )
        for index in range(4):
            rows.append(
                {
                    "row_key": f"train:metal:{index}",
                    "composition_valid": True,
                    "composition_reason": "all_metal_shortcut",
                }
            )
        for index in range(6):
            rows.append(
                {
                    "row_key": f"train:invalid:{index}",
                    "composition_valid": False,
                    "composition_reason": "charge_neutrality_fail",
                }
            )
        stream = build_training_stream(rows, total_rows=10, positive_fraction=0.8, seed=17)
        positives = [row for row in stream if row["stream_role"] == "chemistry_valid_positive"]
        # Source all-metal share is 4/20, so floor(8 * .2) = 1.
        self.assertEqual(
            sum(row["composition_reason"] == "all_metal_shortcut" for row in positives),
            1,
        )

    def test_validation_panel_is_hash_fixed(self):
        rows = [{"row_key": f"val:{index}"} for index in range(20)]
        first = build_validation_panel(rows, count=8, seed=17)
        second = build_validation_panel(rows, count=8, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_joint_donor_attachment_is_deterministic(self):
        source_plans = [
            plan("Li2O", ["Li", "O"], [2, 1]),
            plan(
                "MgO2",
                ["Mg", "O"],
                [1, 2],
                lattice="hexagonal",
                spacegroup="sg_168_194",
                volume="volpa_005_009",
            ),
        ]
        rows = [
            {
                "row_key": f"train:{index}",
                "_plan": value,
                "composition_valid": True,
            }
            for index, value in enumerate(source_plans)
        ]
        attach_joint_negatives(rows, seed=17)
        self.assertTrue(all(row["joint_negative_audit"]["available"] for row in rows))
        self.assertIn("lattice: hexagonal", rows[0]["joint_negative_answer"])

    def test_enrichment_and_written_output_drop_internal_plan(self):
        source_plan = plan("Li2O", ["Li", "O"], [2, 1])
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "prompt": None,
            "answer": answer(source_plan),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            enriched = enrich_rows(
                path,
                split="train",
                seed=17,
                classifier=synthetic_classifier,
            )
            self.assertEqual(len(enriched), 1)
            self.assertTrue(enriched[0]["composition_valid"])
            self.assertIn("_plan", enriched[0])
            public = public_row(enriched[0])
            self.assertNotIn("_plan", public)
            output = Path(tmp) / "out.jsonl"
            write_jsonl(output, enriched)
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn('"_plan"', serialized)

    def test_leakage_firewall_rejects_energy_or_sun_fields(self):
        with self.assertRaises(ValueError):
            assert_no_leakage({"e_hull": 0.0})
        with self.assertRaises(ValueError):
            assert_no_leakage({"strict_sun": True})
        assert_no_leakage({"formula": "Li2O", "composition_valid": True})

    def test_proportional_anchor_selection_preserves_dominant_stratum(self):
        rows = []
        for index in range(90):
            rows.append(
                {
                    "row_key": f"train:oxide:{index}",
                    "N": 8,
                    "arity": 2,
                    "anion_framework": "oxide",
                    "charge_bucket": "neutral_plausible",
                    "lattice_system": "cubic",
                    "spacegroup_bucket": "sg_195_230",
                    "volume_per_atom_bin": "volpa_010_014",
                }
            )
        for index in range(10):
            rows.append(
                {
                    "row_key": f"train:halide:{index}",
                    "N": 12,
                    "arity": 3,
                    "anion_framework": "halide",
                    "charge_bucket": "neutral_plausible",
                    "lattice_system": "orthorhombic",
                    "spacegroup_bucket": "sg_016_074",
                    "volume_per_atom_bin": "volpa_015_019",
                }
            )
        selected = _proportional_stratified_selection(
            rows,
            count=20,
            seed=17,
            namespace="test-anchor",
        )
        self.assertEqual(
            sum(row["anion_framework"] == "oxide" for row in selected),
            18,
        )

    def test_target_mask_excludes_prompt(self):
        labels = [-100, -100, -100, 10, 11, 12]
        self.assertEqual(supervised_target_positions(labels), (3, 4, 5))

    def test_target_only_tokenization_is_prompt_perturbation_invariant(self):
        tokenizer = FakeTokenizer()
        first = build_target_only_token_lists(tokenizer, "short prompt", "answer", 64)
        second = build_target_only_token_lists(
            tokenizer,
            "a different and much longer prompt",
            "answer",
            64,
        )
        first_targets = [
            value for value in first["labels"] if value != -100
        ]
        second_targets = [
            value for value in second["labels"] if value != -100
        ]
        self.assertEqual(first_targets, second_targets)
        self.assertEqual(first_targets, first["input_ids"][-len(first_targets) :])
        self.assertTrue(
            all(value == -100 for value in first["labels"][: first["prompt_token_count"]])
        )
        self.assertEqual(
            len(supervised_target_positions(first["labels"])),
            first["target_token_count"],
        )

    def test_invalid_rows_do_not_receive_joint_negatives(self):
        source = {
            "row_key": "train:invalid",
            "_plan": plan("Li2O", ["Li", "O"], [2, 1]),
            "composition_valid": False,
        }
        donor = {
            "row_key": "train:valid",
            "_plan": plan(
                "MgO2",
                ["Mg", "O"],
                [1, 2],
                lattice="hexagonal",
                spacegroup="sg_168_194",
                volume="volpa_005_009",
            ),
            "composition_valid": True,
        }
        rows = [source, donor]
        attach_joint_negatives(rows, seed=17)
        self.assertIsNone(source["joint_negative_answer"])
        self.assertEqual(
            source["joint_negative_audit"]["reason"],
            "positive_not_composition_valid",
        )

    def test_same_formula_polymorph_is_not_used_as_joint_negative(self):
        source_plan = plan("Li2O", ["Li", "O"], [2, 1])
        polymorph_plan = plan(
            "Li2O",
            ["Li", "O"],
            [2, 1],
            lattice="hexagonal",
            spacegroup="sg_168_194",
            volume="volpa_005_009",
        )
        other_plan = plan(
            "MgO2",
            ["Mg", "O"],
            [1, 2],
            lattice="orthorhombic",
            spacegroup="sg_016_074",
            volume="volpa_015_019",
        )
        rows = [
            {
                "row_key": "train:source",
                "_plan": source_plan,
                "composition_valid": True,
            },
            {
                "row_key": "train:polymorph",
                "_plan": polymorph_plan,
                "composition_valid": True,
            },
            {
                "row_key": "train:other",
                "_plan": other_plan,
                "composition_valid": True,
            },
        ]
        attach_joint_negatives(rows, seed=17)
        self.assertEqual(rows[0]["joint_negative_audit"]["donor_formula"], "MgO2")

    def test_same_reduced_formula_is_not_used_as_joint_negative(self):
        source_plan = plan("LiO", ["Li", "O"], [1, 1])
        scaled_same_formula = plan(
            "Li2O2",
            ["Li", "O"],
            [2, 2],
            lattice="hexagonal",
            spacegroup="sg_168_194",
            volume="volpa_005_009",
        )
        other_plan = plan(
            "MgO2",
            ["Mg", "O"],
            [1, 2],
            lattice="orthorhombic",
            spacegroup="sg_016_074",
            volume="volpa_015_019",
        )
        rows = [
            {
                "row_key": "train:source-reduced",
                "_plan": source_plan,
                "composition_valid": True,
            },
            {
                "row_key": "train:scaled-same-formula",
                "_plan": scaled_same_formula,
                "composition_valid": True,
            },
            {
                "row_key": "train:other-reduced",
                "_plan": other_plan,
                "composition_valid": True,
            },
        ]
        attach_joint_negatives(rows, seed=17)
        self.assertEqual(rows[0]["joint_negative_audit"]["donor_formula"], "MgO2")


if __name__ == "__main__":
    unittest.main()

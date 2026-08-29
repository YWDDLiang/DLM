import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_d3po_listwise_groups",
    ROOT / "scripts/build_d3po_listwise_groups.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

try:
    import pymatgen  # noqa: F401
except ModuleNotFoundError:
    pymatgen = None


def plan(elements, counts, *, family="oxide"):
    return {
        "N": sum(counts),
        "elements": list(elements),
        "counts": list(counts),
        "anion_framework": family,
        "charge_bucket": "neutral_plausible",
    }


def outcome(plan_state, answer, energy, source):
    prompt, reason = MODULE.PAIR_BUILDER.minimal_prompt_from_plan(plan_state)
    assert reason == "ok" and prompt is not None
    return {
        "answer": answer,
        "cif": f"cif-for-{answer}",
        "energy_per_atom": energy,
        "prompt": prompt,
        "plan": dict(plan_state),
        "source": source,
        "source_ordinal": 0,
    }


class BuildD3POListwiseGroupsTest(unittest.TestCase):
    def test_one_weight_one_row_per_exact_composition(self):
        state = plan(["Li", "O"], [2, 1])
        identity = MODULE.PAIR_BUILDER.composition_identity(state)
        groups = {
            identity: [
                outcome(state, "same", -2.00, "s0"),
                outcome(state, "same", -1.90, "s1"),
                outcome(state, "other", -1.80, "s2"),
            ]
        }
        rows_by_split, audit = MODULE.build_listwise_rows(
            groups,
            physical_deduplication=False,
        )
        rows = rows_by_split["train"] + rows_by_split["validation"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["composition_id"], "Li:2|O:1")
        self.assertEqual(row["group_weight"], 1.0)
        self.assertEqual(row["candidate_count"], 2)
        self.assertEqual(len(row["candidates"]), 2)
        self.assertEqual(row["energy_label"], MODULE.ENERGY_LABEL)
        self.assertAlmostEqual(
            row["candidates"][0]["post_model494_energy_per_atom"],
            -1.95,
        )
        self.assertEqual(row["candidates"][0]["replicate_count"], 2)
        self.assertEqual(audit["deduplication"]["exact_duplicates_removed"], 1)

    def test_k_less_than_two_is_excluded(self):
        state = plan(["Na", "O"], [2, 1])
        identity = MODULE.PAIR_BUILDER.composition_identity(state)
        rows_by_split, audit = MODULE.build_listwise_rows(
            {identity: [outcome(state, "only", -1.0, "s0")]},
            physical_deduplication=False,
        )
        self.assertEqual(rows_by_split, {"train": [], "validation": []})
        self.assertEqual(
            audit["skipped_groups"]["fewer_than_two_physical_candidates"],
            1,
        )

    def test_chemsys_is_disjoint_between_splits(self):
        groups = {}
        elements = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr"]
        for element in elements:
            state = plan([element, "O"], [2, 1])
            identity = MODULE.PAIR_BUILDER.composition_identity(state)
            groups[identity] = [
                outcome(state, f"{element}-a", -2.0, "s0"),
                outcome(state, f"{element}-b", -1.9, "s1"),
            ]
        rows_by_split, _ = MODULE.build_listwise_rows(
            groups,
            physical_deduplication=False,
        )
        train = {row["chemsys"] for row in rows_by_split["train"]}
        validation = {
            row["chemsys"] for row in rows_by_split["validation"]
        }
        self.assertFalse(train & validation)
        self.assertEqual(
            len(rows_by_split["train"]) + len(rows_by_split["validation"]),
            len(groups),
        )

    def test_write_dataset_records_hashes_and_holdout_guards(self):
        state = plan(["Li", "O"], [2, 1])
        identity = MODULE.PAIR_BUILDER.composition_identity(state)
        rows_by_split, audit = MODULE.build_listwise_rows(
            {
                identity: [
                    outcome(state, "a", -2.0, "s0"),
                    outcome(state, "b", -1.8, "s1"),
                ]
            },
            physical_deduplication=False,
        )
        source_records = [
            {
                "role": "fixture",
                "path": "/frozen/fixture.jsonl",
                "bytes": 7,
                "sha256": "a" * 64,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "asset"
            returned = MODULE.write_dataset(
                output,
                rows_by_split,
                source_records=source_records,
                audit=audit,
            )
            manifest = json.loads(
                (output / "D3PO_LISTWISE_GROUP_MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(manifest["main_holdout_outcomes_read"])
            self.assertFalse(manifest["sealed_holdout_outcomes_read"])
            self.assertEqual(manifest["source_files"], source_records)
            self.assertEqual(manifest["group_weight"], 1.0)
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(len(returned["manifest_sha256"]), 64)
            for split, digest in manifest["output_hashes"].items():
                self.assertEqual(
                    MODULE.sha256_file(output / f"{split}.jsonl"), digest
                )

    def test_cli_has_no_holdout_outcome_argument(self):
        option_strings = {
            option
            for action in MODULE.build_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(
            any("holdout" in option.lower() for option in option_strings)
        )

    @unittest.skipIf(pymatgen is None, "pymatgen is not installed")
    def test_physical_equivalents_are_collapsed(self):
        state = plan(["Li", "O"], [1, 1])
        identity = MODULE.PAIR_BUILDER.composition_identity(state)
        cif_a = """data_a
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3
_cell_length_b 3
_cell_length_c 3
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0 0 0
O1 O 0.5 0.5 0.5
"""
        cif_equivalent = cif_a.replace("data_a", "data_b").replace(
            "Li1 Li 0 0 0\nO1 O 0.5 0.5 0.5",
            "O1 O 0.5 0.5 0.5\nLi1 Li 0 0 0",
        )
        cif_distinct = cif_a.replace("data_a", "data_c").replace(
            "_cell_length_a 3\n_cell_length_b 3\n_cell_length_c 3",
            "_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5",
        )
        rows = []
        for answer, cif, energy, source in (
            ("a", cif_a, -2.0, "s0"),
            ("b", cif_equivalent, -1.8, "s1"),
            ("c", cif_distinct, -1.5, "s2"),
        ):
            row = outcome(state, answer, energy, source)
            row["cif"] = cif
            rows.append(row)
        rows_by_split, audit = MODULE.build_listwise_rows(
            {identity: rows},
            physical_deduplication=True,
        )
        groups = rows_by_split["train"] + rows_by_split["validation"]
        self.assertEqual(groups[0]["candidate_count"], 2)
        self.assertEqual(
            audit["deduplication"]["physical_duplicates_removed"], 1
        )


if __name__ == "__main__":
    unittest.main()

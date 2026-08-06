from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1_protocol import (  # noqa: E402
    G1_ATTEMPTS,
    build_seed_ledger,
    evaluate_g1_gate,
    extract_first_json,
    ledger_sha256,
    shuffle_dependency_links,
)


def graph() -> dict:
    return {
        "schema_version": "plangraph_v1",
        "source_plan_state_version": "r5_plan_state_v1",
        "site_group_strategy": "element_multiplicity_v1",
        "composition": {
            "N": 3,
            "elements": ["Na", "Cl"],
            "counts": [1, 2],
            "formula": "NaCl2",
            "reduced_formula": "NaCl2",
            "charge_bucket": "charge_fail",
            "oxidation_candidates": [],
            "anion_framework": "halide",
        },
        "symmetry": {
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
        },
        "lattice": {"volume_per_atom_bin": "volpa_010_014"},
        "site_groups": [
            {
                "group_id": "site_group_000",
                "element": "Na",
                "multiplicity": 1,
                "slot_indices": [0],
                "depends_on": ["composition", "symmetry_lattice"],
            },
            {
                "group_id": "site_group_001",
                "element": "Cl",
                "multiplicity": 2,
                "slot_indices": [1, 2],
                "depends_on": ["composition", "symmetry_lattice"],
            },
        ],
        "constraints": {
            "atom_count": 3,
            "element_counts": {"Na": 1, "Cl": 2},
            "charge_bucket": "charge_fail",
            "composition_locked": True,
        },
        "dependency_order": [
            "composition",
            "symmetry_lattice",
            "site_group_000",
            "site_group_001",
        ],
    }


def report(
    *,
    comp: int,
    parse: int = 512,
    complete: int = 512,
    unique: int = 500,
) -> dict:
    return {
        "attempts": 512,
        "parse_rate": parse / 512,
        "plan_completion_rate": complete / 512,
        "composition_valid_count": comp,
        "composition_valid_rate": comp / 512,
        "unique_formula_count": unique,
        "single_element_rate": 0.01,
        "all_metal_rate": 0.03,
        "num_atoms_histogram": {"4": 256, "8": 256},
        "element_arity_histogram": {"2": 256, "3": 256},
        "lattice_system_histogram": {"cubic": 256, "trigonal": 256},
        "spacegroup_bucket_histogram": {
            "sg_143_167": 256,
            "sg_195_230": 256,
        },
        "anion_framework_histogram": {"oxide": 256, "other": 256},
    }


class G1ProtocolTests(unittest.TestCase):
    def test_seed_ledger_is_exact_and_stable(self) -> None:
        rows = build_seed_ledger()
        self.assertEqual(len(rows), G1_ATTEMPTS)
        self.assertEqual(rows[0], {"ordinal": 0, "seed": 20260731})
        self.assertEqual(rows[-1], {"ordinal": 511, "seed": 20261242})
        self.assertEqual(ledger_sha256(rows), ledger_sha256(build_seed_ledger()))

    def test_shuffle_is_content_keyed_and_preserves_composition(self) -> None:
        source = graph()
        first = shuffle_dependency_links(source, identity="a" * 64)
        repeated = shuffle_dependency_links(source, identity="a" * 64)
        self.assertEqual(first, repeated)
        self.assertEqual(first["composition"], source["composition"])
        self.assertNotEqual(
            first["dependency_order"],
            source["dependency_order"],
        )
        self.assertEqual(set(first), set(source))

    def test_extract_first_json_does_not_repair(self) -> None:
        payload, end = extract_first_json('prefix {"a":1} tail')
        self.assertEqual(payload, {"a": 1})
        self.assertGreater(end, 0)
        with self.assertRaises(ValueError):
            extract_first_json('{"a":')

    def test_gate_pass_and_mechanism_failure(self) -> None:
        reports = {
            "P0": report(comp=470, unique=500),
            "PG": report(comp=492, unique=490),
            "PG-shuffle": report(comp=480, unique=485),
        }
        decision = evaluate_g1_gate(reports)
        self.assertTrue(decision["passed"])
        tied = json.loads(json.dumps(reports))
        tied["PG-shuffle"]["composition_valid_count"] = 492
        self.assertFalse(evaluate_g1_gate(tied)["passed"])


if __name__ == "__main__":
    unittest.main()


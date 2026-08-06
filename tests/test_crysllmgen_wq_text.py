from __future__ import annotations

import dataclasses
import random
import unittest

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.crysllmgen.wq_text import (
    GrammarViolation,
    TopologyEdit,
    audit_synthetic_grammar_transitions,
    crystal_system_for_space_group,
    parse_topology_edit,
    parse_wq_proposal,
    serialize_topology_edit,
    serialize_wq_proposal,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


class FakeCatalog(ChartCatalog):
    def types(self, space_group: int) -> tuple[int, ...]:
        if not 1 <= int(space_group) <= 230:
            raise KeyError(space_group)
        return (0, 1)

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if int(wyckoff_type) == 0:
            return ChartSpec(int(space_group), 0, "a", 1, 0, 1)
        if int(wyckoff_type) == 1:
            return ChartSpec(int(space_group), 1, "b", 2, 1, 2)
        raise KeyError((space_group, wyckoff_type))


def _state() -> StratifiedState:
    return StratifiedState(
        space_group=225,
        lattice_system="cubic",
        lattice_chart=(1.73,),
        orbits=(
            OrbitState("second", 1, 8, 2, 1, (0.375,), 2),
            OrbitState("first", 0, 14, 1, 0, (), 1),
        ),
    )


class CrysLLMGenWyckoffTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = FakeCatalog()

    def test_space_group_crystal_system_boundaries(self) -> None:
        expected = {
            1: "triclinic",
            3: "monoclinic",
            16: "orthorhombic",
            75: "tetragonal",
            143: "trigonal",
            168: "hexagonal",
            195: "cubic",
            230: "cubic",
        }
        self.assertEqual(
            {value: crystal_system_for_space_group(value) for value in expected},
            expected,
        )

    def test_proposal_roundtrip_preserves_presentation_and_support(self) -> None:
        text = serialize_wq_proposal(_state(), self.catalog)
        self.assertRegex(
            text,
            r"^SG=225;Q=[0-9A-F]{2};O=0,W=1,E=8,U=[0-9A-F]{2};"
            r"O=1,W=0,E=14,U=-;STOP$",
        )
        recovered = parse_wq_proposal(text, self.catalog, attempt_id="attempt-0")
        self.assertEqual(recovered.attempt_id, "attempt-0")
        self.assertEqual(recovered.atom_count, 3)
        self.assertEqual(
            [(orbit.wyckoff_type, orbit.species) for orbit in recovered.orbits],
            [(1, 8), (0, 14)],
        )
        self.assertLess(abs(recovered.orbits[0].free_coordinate[0] - 0.375), 1 / 256)

    def test_species_field_is_mp20_class_not_truncated_atomic_number(self) -> None:
        state = dataclasses.replace(
            _state(),
            orbits=(dataclasses.replace(_state().orbits[1], species=94),),
        )
        text = serialize_wq_proposal(state, self.catalog)
        self.assertIn("E=89", text)
        recovered = parse_wq_proposal(text, self.catalog)
        self.assertEqual(recovered.orbits[0].species, 94)

    def test_random_orbit_presentation_changes_text_not_canonical_topology(self) -> None:
        state = _state()
        permuted = state.permuted(random.Random(1))
        self.assertNotEqual(
            serialize_wq_proposal(state, self.catalog),
            serialize_wq_proposal(permuted, self.catalog),
        )
        self.assertEqual(state.topology_hash(), permuted.topology_hash())

    def test_parser_never_repairs_prose_whitespace_or_illegal_support(self) -> None:
        valid = serialize_wq_proposal(_state(), self.catalog)
        for value in (
            "answer=" + valid,
            valid + "\n",
            valid.replace("W=1", "W=9", 1),
            valid.replace("O=1", "O=3", 1),
            valid.replace("U=-", "U=00", 1),
        ):
            with self.assertRaises(GrammarViolation):
                parse_wq_proposal(value, self.catalog)

    def test_parser_rejects_more_than_twenty_primitive_atoms(self) -> None:
        fields = ["SG=225", "Q=80"]
        fields.extend(f"O={index},W=0,E=1,U=-" for index in range(21))
        fields.append("STOP")
        with self.assertRaisesRegex(GrammarViolation, "MP20"):
            parse_wq_proposal(";".join(fields), self.catalog)

    def test_direct_edit_grammar_masks_invalid_events(self) -> None:
        state = _state()
        self.assertEqual(parse_topology_edit("NOOP", state, self.catalog).kind, "noop")
        edit = parse_topology_edit("TYPE;O=1;W=1", state, self.catalog)
        self.assertEqual(edit, TopologyEdit("type_change", orbit_index=1, wyckoff_type=1))
        self.assertEqual(serialize_topology_edit(edit), "TYPE;O=1;W=1")
        with self.assertRaises(GrammarViolation):
            parse_topology_edit("TYPE;O=1;W=0", state, self.catalog)
        with self.assertRaises(GrammarViolation):
            parse_topology_edit("SPECIES;O=0;E=8", state, self.catalog)
        singleton = dataclasses.replace(state, orbits=(state.orbits[0],))
        with self.assertRaisesRegex(GrammarViolation, "empty"):
            parse_topology_edit("DEATH;O=0", singleton, self.catalog)

    def test_synthetic_transition_sampler_has_zero_illegal_support(self) -> None:
        report = audit_synthetic_grammar_transitions(
            self.catalog, transitions=10_000, seed=20260720
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["illegal_generated"], 0)
        self.assertEqual(sum(report["phase_counts"].values()), 10_000)


if __name__ == "__main__":
    unittest.main()

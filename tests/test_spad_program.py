import unittest

from crystal_dlm.spad_program import (
    anchor_revision_slots,
    begin_anchor_revision,
    canonical_predictor_position_groups,
    coordinate_positions,
    program_from_element_order,
    program_from_planner_trace,
    spad_predictor_position_groups,
)


class SPADProgramTest(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "N": 6,
            "elements": ["O", "Na", "Cl"],
            "counts": [3, 1, 2],
        }

    def test_external_llama_order_maps_to_canonical_noncontiguous_slots(self):
        program = program_from_element_order(
            self.plan,
            ["O", "Na", "Cl"],
            order_source="llama_program_head",
        )
        # Canonical storage is O(0..2), Na(3), Cl(4..5), but O -> Na -> Cl
        # remains explicit runtime metadata.
        self.assertEqual(program.element_order, ("O", "Na", "Cl"))
        self.assertEqual(program.anchor_slots, (0, 3, 4))
        self.assertEqual(program.entries[0].remaining_slots, (1, 2))

    def test_trace_merges_oxidation_variants_and_checks_exact_composition(self):
        trace = [
            {"action": "proposal", "N": 6},
            {"action": "species", "atomic_number": 8, "oxidation_state": -2, "count": 2},
            {"action": "species", "atomic_number": 8, "oxidation_state": -1, "count": 1},
            {"action": "species", "atomic_number": 11, "oxidation_state": 1, "count": 1},
            {"action": "species", "atomic_number": 17, "oxidation_state": -1, "count": 2},
            {"action": "EOS"},
        ]
        program = program_from_planner_trace(self.plan, trace)
        self.assertEqual(program.element_order, ("O", "Na", "Cl"))
        self.assertEqual([entry.count for entry in program.entries], [3, 1, 2])

    def test_trace_rejects_plan_mismatch_and_missing_eos(self):
        with self.assertRaisesRegex(ValueError, "lacks EOS"):
            program_from_planner_trace(
                self.plan,
                [{"action": "species", "atomic_number": 8, "count": 3}],
            )
        with self.assertRaisesRegex(ValueError, "composition disagrees"):
            program_from_planner_trace(
                self.plan,
                [
                    {"action": "species", "atomic_number": 8, "count": 2},
                    {"action": "EOS"},
                ],
            )

    def test_predictor_schedule_covers_7_plus_4n_once_and_is_future_first(self):
        program = program_from_element_order(
            self.plan,
            ["Cl", "O", "Na"],
            order_source="llama_program_head",
        )
        groups = spad_predictor_position_groups(program)
        flattened = [position for group in groups for position in group]
        self.assertEqual(len(flattened), 7 + 4 * self.plan["N"])
        self.assertEqual(set(flattened), set(range(7 + 4 * self.plan["N"])))
        # Cl anchor is native slot 4 and is generated before the O anchor at 0.
        self.assertLess(groups.index((coordinate_positions(4)[0],)), groups.index((coordinate_positions(0)[0],)))
        # O's later native slot 1 is deferred until every species has an anchor.
        self.assertLess(groups.index((coordinate_positions(3)[0],)), groups.index((coordinate_positions(1)[0],)))

    def test_canonical_control_is_same_canvas_but_native_site_order(self):
        groups = canonical_predictor_position_groups(3)
        self.assertLess(groups.index((8,)), groups.index((12,)))
        self.assertLess(groups.index((12,)), groups.index((16,)))

    def test_anchor_revision_preserves_old_xyz_for_geometry_and_suffix(self):
        values = tuple(range(31))
        revision = begin_anchor_revision(
            values,
            slot_index=3,
            mask_token_id=999,
            suffix_visible=True,
        )
        self.assertEqual(revision.positions, coordinate_positions(3))
        self.assertEqual(
            tuple(revision.masked_token_ids[pos] for pos in revision.positions),
            (999, 999, 999),
        )
        self.assertEqual(revision.provisional_token_ids(), values)
        self.assertTrue(all(revision.visible_positions))

    def test_no_suffix_diagnostic_changes_visibility_not_state(self):
        values = tuple(range(31))
        visible = begin_anchor_revision(
            values, slot_index=1, mask_token_id=999, suffix_visible=True
        )
        hidden = begin_anchor_revision(
            values, slot_index=1, mask_token_id=999, suffix_visible=False
        )
        self.assertEqual(visible.masked_token_ids, hidden.masked_token_ids)
        self.assertTrue(all(visible.visible_positions))
        self.assertFalse(hidden.visible_positions[-1])
        self.assertTrue(hidden.visible_positions[coordinate_positions(1)[-1]])

    def test_reverse_program_order_defines_one_revision_per_species(self):
        program = program_from_element_order(
            self.plan,
            ["Cl", "O", "Na"],
            order_source="llama_program_head",
        )
        self.assertEqual(anchor_revision_slots(program), (3, 0, 4))

    def test_program_must_be_exact_unique_permutation(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            program_from_element_order(
                self.plan, ["O", "O", "Na"], order_source="test"
            )
        with self.assertRaisesRegex(ValueError, "permute"):
            program_from_element_order(
                self.plan, ["O", "Na"], order_source="test"
            )


if __name__ == "__main__":
    unittest.main()

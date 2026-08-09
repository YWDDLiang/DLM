from copy import deepcopy
import random
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.plangraph_v1 import plangraph_from_plan_state
from crystal_dlm.planned_corruption import (
    CorruptionScheduleError,
    PositionGroup,
    corruption_key_for_record,
    current_order_groups,
    h1a2_generation_schedule,
    plangraph_dependency_groups,
    position_group_ids,
    sample_iid_corruption,
    sample_mixture_policy,
    sample_planned_corruption,
    safe_axis_dependency_groups,
    simulate_planned_policy,
    stateless_uniform,
    validate_position_groups,
)
from crystal_dlm.r5_dynamic_length import exact_dynamic_generation_schedule
from crystal_dlm.r5_plan_state import plan_state_from_arrays


class PlannedCorruptionTests(unittest.TestCase):
    def make_graph(self):
        answer, _diagnostics = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(
            arrays,
            metadata={"spacegroup.number": 194},
        )
        return plangraph_from_plan_state(
            plan,
            site_species=arrays["species"],
        )

    def test_d1_groups_match_current_generation_schedule(self):
        groups = current_order_groups(3)
        existing = exact_dynamic_generation_schedule(3)
        self.assertEqual(
            [list(group.positions) for group in groups],
            existing,
        )
        validate_position_groups(groups, answer_length=19)

    def test_d2_groups_cover_every_position_once(self):
        groups = plangraph_dependency_groups(self.make_graph())
        self.assertEqual(
            [group.name for group in groups],
            [
                "composition",
                "symmetry_lattice",
                "site_group_000",
                "site_group_001",
            ],
        )
        self.assertEqual(groups[0].positions, (0, 7, 11, 15))
        self.assertEqual(groups[1].positions, (1, 2, 3, 4, 5, 6))
        self.assertEqual(
            groups[2].positions,
            (8, 9, 10, 16, 17, 18),
        )
        self.assertEqual(groups[3].positions, (12, 13, 14))
        validate_position_groups(groups, answer_length=19)
        encoded = position_group_ids(groups, answer_length=19)
        self.assertEqual(encoded[0], 0)
        self.assertEqual(encoded[7], 0)
        self.assertEqual(encoded[1:7], (1, 1, 1, 1, 1, 1))
        self.assertEqual(encoded[8:11], (2, 2, 2))
        self.assertEqual(encoded[12:15], (3, 3, 3))

    def test_safe_axis_groups_are_axis_pure_and_put_all_z_last(self):
        groups = safe_axis_dependency_groups(self.make_graph())
        names = [group.name for group in groups]
        self.assertEqual(names[:2], ["composition", "symmetry_lattice"])
        x_indices = [index for index, name in enumerate(names) if name.endswith("_x")]
        y_indices = [index for index, name in enumerate(names) if name.endswith("_y")]
        z_indices = [index for index, name in enumerate(names) if name.endswith("_z")]
        self.assertTrue(x_indices and y_indices and z_indices)
        self.assertLess(max([*x_indices, *y_indices]), min(z_indices))
        for group in groups[2:]:
            axes = {(position - 7) % 4 for position in group.positions}
            self.assertEqual(len(axes), 1)
        validate_position_groups(groups, answer_length=19)

    def test_planned_mask_keeps_prerequisites_and_masks_future(self):
        groups = plangraph_dependency_groups(self.make_graph())
        sample = sample_planned_corruption(
            groups,
            rng=random.Random(7),
            active_group_index=2,
            p_mask=0.25,
            policy_name="d2",
        )
        prerequisites = {
            position for group in groups[:2] for position in group.positions
        }
        active = set(groups[2].positions)
        future = set(groups[3].positions)

        self.assertFalse(prerequisites.intersection(sample.masked_input_positions))
        self.assertTrue(future.issubset(sample.masked_input_positions))
        self.assertTrue(set(sample.loss_positions).issubset(active))
        self.assertTrue(sample.loss_positions)
        self.assertFalse(future.intersection(sample.loss_positions))

    def test_planned_mask_is_seed_deterministic(self):
        groups = current_order_groups(4)
        first = sample_planned_corruption(
            groups,
            rng=random.Random(123),
            policy_name="d1",
        )
        second = sample_planned_corruption(
            groups,
            rng=random.Random(123),
            policy_name="d1",
        )
        self.assertEqual(first, second)

    def test_active_group_always_contributes_loss(self):
        groups = current_order_groups(2)
        sample = sample_planned_corruption(
            groups,
            rng=random.Random(1),
            active_group_index=0,
            p_mask=1e-12,
        )
        self.assertEqual(sample.loss_positions, (0,))

    def test_iid_masks_and_supervises_the_same_positions(self):
        sample = sample_iid_corruption(
            19,
            rng=random.Random(22),
            p_mask=0.2,
        )
        self.assertEqual(
            sample.masked_input_positions,
            sample.loss_positions,
        )
        self.assertTrue(sample.loss_positions)

    def test_registered_two_to_one_mixture_is_reproducible(self):
        rng = random.Random(20260731)
        draws = [
            sample_mixture_policy(
                rng=rng,
                iid_weight=2,
                planned_weight=1,
            )
            for _ in range(6000)
        ]
        planned = draws.count("planned")
        self.assertGreater(planned, 1900)
        self.assertLess(planned, 2100)
        repeated_rng = random.Random(20260731)
        repeated = [
            sample_mixture_policy(
                rng=repeated_rng,
                iid_weight=2,
                planned_weight=1,
            )
            for _ in range(6000)
        ]
        self.assertEqual(draws, repeated)

    def test_corruption_key_uses_pair_content_not_metadata_or_row_id(self):
        first = {
            "prompt": "plan\n",
            "answer": "<N_001>",
            "material_id": "secret-row-a",
            "metadata": {"energy": -1.0},
        }
        second = {
            "prompt": "plan\n",
            "answer": "<N_001>",
            "material_id": "secret-row-b",
            "metadata": {"energy": 99.0},
        }
        self.assertEqual(
            corruption_key_for_record(first),
            corruption_key_for_record(second),
        )
        second["answer"] = "<N_002>"
        self.assertNotEqual(
            corruption_key_for_record(first),
            corruption_key_for_record(second),
        )

    def test_h1a2_generation_schedules_are_inference_available(self):
        graph = self.make_graph()
        plan = {
            **graph["composition"],
            "lattice_system": graph["symmetry"]["lattice_system"],
            "spacegroup_bucket": graph["symmetry"]["spacegroup_bucket"],
            "volume_per_atom_bin": graph["lattice"]["volume_per_atom_bin"],
        }
        d1 = h1a2_generation_schedule(plan, policy="d1")
        d2 = h1a2_generation_schedule(plan, policy="d2")
        safe_axis = h1a2_generation_schedule(plan, policy="d2_safe_axis")
        self.assertEqual(d1, exact_dynamic_generation_schedule(plan["N"]))
        self.assertEqual(
            sorted(position for group in d2 for position in group),
            list(range(7 + 4 * int(plan["N"]))),
        )
        self.assertEqual(
            safe_axis,
            [list(group.positions) for group in safe_axis_dependency_groups(graph)],
        )
        with self.assertRaises(CorruptionScheduleError):
            h1a2_generation_schedule(plan, policy="d2_shuffle")

    def test_stateless_uniform_is_counter_deterministic(self):
        value = stateless_uniform(
            123456789,
            step=17,
            seed=20260731,
            stream=2,
            position=9,
        )
        repeated = stateless_uniform(
            123456789,
            step=17,
            seed=20260731,
            stream=2,
            position=9,
        )
        self.assertEqual(value, repeated)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        changed = {
            stateless_uniform(
                123456789,
                step=17,
                seed=20260731,
                stream=stream,
                position=position,
            )
            for stream, position in ((2, 10), (3, 9), (3, 10))
        }
        self.assertEqual(len(changed), 3)

    def test_stateless_primary_mixture_is_close_to_two_to_one(self):
        planned = sum(
            stateless_uniform(
                key,
                step=key % 37,
                seed=20260731,
                stream=0,
            )
            < (1.0 / 3.0)
            for key in range(1, 6001)
        )
        self.assertGreater(planned, 1900)
        self.assertLess(planned, 2100)

    def test_simulation_reports_every_group(self):
        groups = plangraph_dependency_groups(self.make_graph())
        summary = simulate_planned_policy(
            groups,
            trials=2000,
            seed=9,
            policy_name="d2",
        )
        self.assertEqual(summary["answer_length"], 19)
        self.assertEqual(sum(summary["active_group_counts"].values()), 2000)
        self.assertTrue(
            all(count > 0 for count in summary["active_group_counts"].values())
        )
        self.assertEqual(len(summary["loss_frequency"]), 19)
        self.assertTrue(all(value > 0.0 for value in summary["loss_frequency"]))

    def test_group_validation_rejects_overlap(self):
        groups = (
            PositionGroup("first", (0, 1), ()),
            PositionGroup("second", (1, 2), ("first",)),
        )
        with self.assertRaises(CorruptionScheduleError):
            validate_position_groups(groups, answer_length=3)

    def test_invalid_plangraph_is_rejected_before_grouping(self):
        graph = deepcopy(self.make_graph())
        graph["site_groups"][0]["slot_indices"] = [0, 1, 2]
        with self.assertRaises(Exception):
            plangraph_dependency_groups(graph)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from crystal_dlm.wqcodiff.crysllmgen.bridge_parity import (
    ATTEMPTS_PER_TIMESTEP,
    BRIDGE_CELL_COUNT,
    BRIDGE_TIMESTEPS,
    CleanProposalCondition,
    ParentScheduleArrays,
    build_bridge_cells,
    build_numpy_parent_schedules,
    forward_noise_numpy,
    reconstruction_errors,
    respaced_timesteps,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_schedule_correct_bridge_parity_v1.json"
)
AUTHORIZATION = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_schedule_correct_bridge_parity_v1"
    / "authorization_record.json"
)


class ScheduleCorrectBridgeParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedules = build_numpy_parent_schedules()
        self.condition = CleanProposalCondition(
            frac_coords=np.asarray(
                [
                    [0.0, 0.25, 0.5],
                    [0.125, 0.375, 0.625],
                    [0.75, 0.875, 0.0625],
                ],
                dtype=np.float64,
            ),
            lattice=np.asarray(
                [
                    [4.1, 0.0, 0.0],
                    [0.2, 4.3, 0.0],
                    [0.1, 0.3, 5.2],
                ],
                dtype=np.float64,
            ),
        )
        self.cells = build_bridge_cells(base_seed=2026072601)

    def test_exact_4_by_8_matrix_and_paired_noise(self) -> None:
        self.assertEqual(len(self.cells), BRIDGE_CELL_COUNT)
        self.assertEqual(len({cell.cell_id for cell in self.cells}), BRIDGE_CELL_COUNT)
        for timestep in BRIDGE_TIMESTEPS:
            selected = [cell for cell in self.cells if cell.timestep == timestep]
            self.assertEqual(len(selected), ATTEMPTS_PER_TIMESTEP)
            self.assertEqual(
                [cell.panel_index for cell in selected],
                list(range(ATTEMPTS_PER_TIMESTEP)),
            )
        for panel_index in range(ATTEMPTS_PER_TIMESTEP):
            paired = [cell for cell in self.cells if cell.panel_index == panel_index]
            self.assertEqual(len({cell.forward_noise_seed for cell in paired}), 1)
            self.assertEqual(len({cell.reverse_noise_seed for cell in paired}), 1)

    def test_forward_noise_matches_parent_formula_and_reconstructs(self) -> None:
        original_coordinates = np.array(self.condition.frac_coords, copy=True)
        original_lattice = np.array(self.condition.lattice, copy=True)
        for cell in self.cells:
            bridge_input = forward_noise_numpy(
                self.condition,
                schedules=self.schedules,
                cell=cell,
            )
            state = bridge_input.state
            expected_coordinates = (
                self.condition.frac_coords
                + state.coordinate_sigma * state.coordinate_noise
            ) % 1.0
            expected_lattice = (
                np.sqrt(state.alpha_bar) * self.condition.lattice
                + np.sqrt(1.0 - state.alpha_bar) * state.lattice_noise
            )
            np.testing.assert_allclose(state.frac_coords, expected_coordinates)
            np.testing.assert_allclose(state.lattice, expected_lattice)
            errors = reconstruction_errors(bridge_input)
            self.assertLessEqual(
                errors["coordinate_periodic_max_abs_error"], 1.0e-12
            )
            self.assertLessEqual(errors["lattice_max_abs_error"], 1.0e-12)
            self.assertFalse(
                np.shares_memory(
                    bridge_input.condition.frac_coords, bridge_input.state.frac_coords
                )
            )
            self.assertFalse(
                np.shares_memory(
                    bridge_input.condition.lattice, bridge_input.state.lattice
                )
            )
        np.testing.assert_array_equal(self.condition.frac_coords, original_coordinates)
        np.testing.assert_array_equal(self.condition.lattice, original_lattice)

    def test_condition_and_state_are_read_only(self) -> None:
        bridge_input = forward_noise_numpy(
            self.condition,
            schedules=self.schedules,
            cell=self.cells[0],
        )
        with self.assertRaises(ValueError):
            bridge_input.condition.frac_coords[0, 0] = 0.5
        with self.assertRaises(ValueError):
            bridge_input.state.lattice[0, 0] = 0.5

    def test_schedule_is_1000_steps_and_monotone(self) -> None:
        self.assertEqual(self.schedules.alphas_cumprod.shape, (1001,))
        self.assertEqual(self.schedules.coordinate_sigmas.shape, (1001,))
        self.assertEqual(float(self.schedules.alphas_cumprod[0]), 1.0)
        self.assertEqual(float(self.schedules.coordinate_sigmas[0]), 0.0)
        self.assertTrue(np.all(np.diff(self.schedules.alphas_cumprod) <= 0.0))
        self.assertTrue(np.all(np.diff(self.schedules.coordinate_sigmas) >= 0.0))

    def test_mutated_schedule_and_matrix_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_numpy_parent_schedules(timesteps=800)
        with self.assertRaises(ValueError):
            build_bridge_cells(base_seed=1, timesteps=(100, 200, 400))
        with self.assertRaises(ValueError):
            build_bridge_cells(base_seed=1, attempts_per_timestep=7)
        with self.assertRaises(ValueError):
            ParentScheduleArrays(
                alphas_cumprod=self.schedules.alphas_cumprod[:-1],
                coordinate_sigmas=self.schedules.coordinate_sigmas,
            )

    def test_invalid_clean_geometry_fails_before_noise(self) -> None:
        with self.assertRaises(ValueError):
            CleanProposalCondition(
                frac_coords=np.zeros((2, 3)),
                lattice=np.zeros((3, 3)),
            )
        with self.assertRaises(ValueError):
            CleanProposalCondition(
                frac_coords=np.asarray([[1.0, 0.0, 0.0]]),
                lattice=np.eye(3),
            )
        with self.assertRaises(FloatingPointError):
            CleanProposalCondition(
                frac_coords=np.asarray([[np.nan, 0.0, 0.0]]),
                lattice=np.eye(3),
            )

    def test_respaced_grids_are_exact_and_descending(self) -> None:
        for timestep in BRIDGE_TIMESTEPS:
            grid = respaced_timesteps(timestep)
            self.assertEqual(len(grid), 32)
            self.assertEqual(grid[0], timestep)
            self.assertEqual(grid[-1], 1)
            self.assertEqual(len(set(grid)), 32)
            self.assertTrue(all(first > second for first, second in zip(grid, grid[1:])))

    def test_contract_preserves_fail_and_forbids_automatic_training(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        self.assertEqual(contract["non_rewrite_boundary"]["formal_all22_gate"], "FAIL")
        self.assertTrue(
            contract["non_rewrite_boundary"][
                "composition_projection_escalation_remains_stopped"
            ]
        )
        self.assertEqual(contract["matrix"]["total_cells"], 32)
        self.assertEqual(contract["matrix"]["timesteps"], list(BRIDGE_TIMESTEPS))
        self.assertEqual(contract["matrix"]["attempts_per_timestep"], 8)
        self.assertTrue(contract["model_selection"]["mlip_free"])
        self.assertFalse(contract["model_selection"]["chgnet_used"])
        self.assertFalse(contract["model_selection"]["mattersim_used"])
        self.assertFalse(contract["model_selection"]["training_performed"])
        self.assertIn(
            "Slurm submission or GPU allocation",
            authorization["interpretation"]["not_authorized_by_this_record"],
        )

    def test_outcome_mutation_cannot_change_cell_identity(self) -> None:
        first = self.cells
        fake_outcomes = {
            cell.cell_id: {"success": bool(index % 2)}
            for index, cell in enumerate(first)
        }
        mutated = copy.deepcopy(fake_outcomes)
        for value in mutated.values():
            value["success"] = not value["success"]
        second = build_bridge_cells(base_seed=2026072601)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.charts import (
    CRYSTAL_SYSTEM_DIMENSIONS,
    LatticeChartCodec,
)
from crystal_dlm.wqcodiff.crysllmgen.tangent_bridge import (
    LatticeChartProjector,
    ManifoldNoiseProjector,
    ParentLatticeFrame,
    TangentProjectionError,
    WyckoffTangentProjector,
    forward_noise_manifold_numpy,
    run_parent_reverse_on_wyckoff_manifold,
    validate_manifold_reverse_identity,
)
from crystal_dlm.wqcodiff.crysllmgen.bridge_parity import (
    build_bridge_cells,
    build_numpy_parent_schedules,
)
from crystal_dlm.wqcodiff.runtime import (
    _PRIMITIVE_TRANSFORM_CACHE,
    expand_state,
)
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


ROOT = Path(__file__).resolve().parents[1]
TANGENT_BRIDGE_SOURCE = (
    ROOT
    / "crystal_dlm"
    / "wqcodiff"
    / "crysllmgen"
    / "tangent_bridge.py"
)


def _triclinic_lattice() -> np.ndarray:
    return np.asarray(
        [
            [4.2, 0.0, 0.0],
            [0.3, 5.1, 0.0],
            [-0.2, 0.4, 6.3],
        ],
        dtype=np.float64,
    )


class _LineCatalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 1)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        return (
            (
                float(free_coordinate[0]) % 1.0,
                0.25,
                0.5,
            ),
        )

    def decode_generator(self, space_group, wyckoff_type, free_coordinate):
        return self.expand(space_group, wyckoff_type, free_coordinate)[0]

    def encode_free(self, space_group, wyckoff_type, position):
        return (float(position[0]) % 1.0,)


class _FixedCatalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 0)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        if tuple(free_coordinate):
            raise ValueError("fixed site cannot receive free coordinates")
        return ((0.0, 0.0, 0.0),)

    def decode_generator(self, space_group, wyckoff_type, free_coordinate):
        return self.expand(space_group, wyckoff_type, free_coordinate)[0]

    def encode_free(self, space_group, wyckoff_type, position):
        return ()


class _RankDeficientCatalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 2)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        # The second declared chart coordinate is intentionally unused.
        return ((float(free_coordinate[0]) % 1.0, 0.0, 0.0),)


class _IllConditionedCatalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 2)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        return (
            (
                float(free_coordinate[0]) % 1.0,
                (1.0e-14 * float(free_coordinate[1])) % 1.0,
                0.0,
            ),
        )


class _CartesianCatalog(ChartCatalog):
    def __init__(self, dimension: int) -> None:
        if dimension not in {2, 3}:
            raise ValueError("synthetic Cartesian catalog supports 2D/3D")
        self.dimension = dimension

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, self.dimension)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        values = tuple(float(value) % 1.0 for value in free_coordinate)
        if len(values) != self.dimension:
            raise ValueError("synthetic chart dimension mismatch")
        if self.dimension == 2:
            return ((values[0], values[1], 0.35),)
        return (values,)

    def decode_generator(self, space_group, wyckoff_type, free_coordinate):
        return self.expand(space_group, wyckoff_type, free_coordinate)[0]

    def encode_free(self, space_group, wyckoff_type, position):
        return tuple(
            float(value) % 1.0 for value in position[: self.dimension]
        )


class _TwoLineCatalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if space_group != 1 or wyckoff_type not in {0, 1}:
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(
            1,
            wyckoff_type,
            "a" if wyckoff_type == 0 else "b",
            1,
            1,
        )

    def types(self, space_group: int):
        return (0, 1)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        value = float(free_coordinate[0]) % 1.0
        return (
            (value, 0.1, 0.2)
            if wyckoff_type == 0
            else (0.3, value, 0.4),
        )

    def decode_generator(self, space_group, wyckoff_type, free_coordinate):
        return self.expand(space_group, wyckoff_type, free_coordinate)[0]

    def encode_free(self, space_group, wyckoff_type, position):
        axis = 0 if wyckoff_type == 0 else 1
        return (float(position[axis]) % 1.0,)


class _CenteredLineCatalog(ChartCatalog):
    """Minimal SG63-like centered orbit used by the job28185 regression."""

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (63, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(63, 0, "a", 2, 1, 1)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        value = float(free_coordinate[0]) % 1.0
        return (
            (value, 0.0, 0.0),
            ((value + 0.5) % 1.0, 0.5, 0.0),
        )

    def decode_generator(self, space_group, wyckoff_type, free_coordinate):
        return self.expand(space_group, wyckoff_type, free_coordinate)[0]

    def encode_free(self, space_group, wyckoff_type, position):
        return (float(position[0]) % 1.0,)


def _state(
    *,
    chart_dimension: int,
    free_coordinate: tuple[float, ...],
) -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=LatticeChartCodec.encode_matrix(
            _triclinic_lattice(),
            "triclinic",
        ),
        orbits=(
            OrbitState(
                orbit_id="orbit-0",
                wyckoff_type=0,
                species=14,
                multiplicity=1,
                chart_dimension=chart_dimension,
                free_coordinate=free_coordinate,
                primitive_multiplicity=1,
            ),
        ),
        attempt_id="unit-test-attempt",
        timestep=0.4,
    )


class LatticeChartProjectorTests(unittest.TestCase):
    def test_parent_lattice_frame_is_a_rigid_roundtrip(self) -> None:
        wq_lattice = _triclinic_lattice()
        right_rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        parent_lattice = wq_lattice @ right_rotation
        frame = ParentLatticeFrame.from_reference_lattices(
            wq_reference_lattice=wq_lattice,
            parent_reference_lattice=parent_lattice,
        )
        np.testing.assert_allclose(
            frame.to_parent(wq_lattice),
            parent_lattice,
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            frame.to_wq(parent_lattice),
            wq_lattice,
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertLessEqual(
            frame.audit.orthogonality_max_abs_error,
            1.0e-12,
        )
        self.assertAlmostEqual(frame.audit.determinant, 1.0)

    def test_every_crystal_system_projects_into_its_registered_chart(self) -> None:
        matrices = {
            "triclinic": np.asarray(
                [[4.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.2, -0.4, 6.0]]
            ),
            "monoclinic": np.asarray(
                [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-1.2, 0.0, 5.8]]
            ),
            "orthorhombic": np.diag([4.0, 5.0, 6.0]),
            "tetragonal": np.diag([4.0, 4.0, 6.0]),
            "hexagonal": np.asarray(
                [
                    [4.0, 0.0, 0.0],
                    [-2.0, 2.0 * math.sqrt(3.0), 0.0],
                    [0.0, 0.0, 6.0],
                ]
            ),
            "trigonal": np.asarray(
                [
                    [4.0, 0.0, 0.0],
                    [-2.0, 2.0 * math.sqrt(3.0), 0.0],
                    [0.0, 0.0, 6.0],
                ]
            ),
            "cubic": np.diag([4.0, 4.0, 4.0]),
        }
        projector = LatticeChartProjector()
        generic_update = np.asarray(
            [
                [0.02, 0.01, -0.01],
                [-0.015, 0.01, 0.02],
                [0.01, -0.02, 0.03],
            ],
            dtype=np.float64,
        )
        for system, matrix in matrices.items():
            with self.subTest(system=system):
                chart = LatticeChartCodec.encode_matrix(matrix, system)
                proposal = matrix + generic_update
                proposal_snapshot = proposal.copy()
                result = projector.project(
                    lattice_chart=chart,
                    lattice_system=system,
                    proposed_conventional_lattice=proposal,
                )
                np.testing.assert_array_equal(proposal, proposal_snapshot)
                self.assertEqual(
                    len(result.lattice_chart),
                    CRYSTAL_SYSTEM_DIMENSIONS[system],
                )
                self.assertEqual(
                    result.audit.jacobian_rank,
                    result.audit.chart_dimension,
                )
                self.assertGreater(
                    result.audit.conventional_volume_after,
                    0.0,
                )
                self.assertLessEqual(
                    result.audit.chart_roundtrip_max_abs_error,
                    1.0e-10,
                )
                self.assertLessEqual(
                    result.audit.lattice_roundtrip_max_abs_error,
                    1.0e-10,
                )
                self.assertTrue(
                    np.all(
                        np.linalg.eigvalsh(
                            result.conventional_lattice
                            @ result.conventional_lattice.T
                        )
                        > 0.0
                    )
                )

    def test_nonfinite_lattice_fails_closed(self) -> None:
        chart = LatticeChartCodec.encode_matrix(
            np.diag([4.0, 4.0, 4.0]),
            "cubic",
        )
        proposal = np.diag([4.0, 4.0, 4.0])
        proposal[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            LatticeChartProjector().project(
                lattice_chart=chart,
                lattice_system="cubic",
                proposed_conventional_lattice=proposal,
            )


class WyckoffTangentProjectorTests(unittest.TestCase):
    def test_job28185_centered_large_step_uses_global_chart_retraction(self) -> None:
        catalog = _CenteredLineCatalog()
        conventional = np.diag(
            [3.784824598541904, 10.7125049330984, 8.337417285599635]
        )
        state = StratifiedState(
            space_group=63,
            lattice_system="orthorhombic",
            lattice_chart=LatticeChartCodec.encode_matrix(
                conventional,
                "orthorhombic",
            ),
            orbits=(
                OrbitState(
                    "o0",
                    0,
                    14,
                    2,
                    1,
                    (0.2,),
                    primitive_multiplicity=1,
                ),
            ),
        )
        transform = np.asarray(
            [
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        previous = _PRIMITIVE_TRANSFORM_CACHE.get(63)
        _PRIMITIVE_TRANSFORM_CACHE[63] = transform.copy()
        try:
            expanded = expand_state(
                state,
                catalog,
                redetect_space_group=False,
            )
            proposed_conventional = conventional.copy()
            proposed_conventional[0, 0] *= 20.0
            result = WyckoffTangentProjector(
                catalog
            ).project_parent_step(
                state=state,
                proposed_fractional_coordinates=(
                    expanded.fractional_coordinates
                ),
                proposed_primitive_lattice=(
                    transform @ proposed_conventional
                ),
            )
        finally:
            if previous is None:
                _PRIMITIVE_TRANSFORM_CACHE.pop(63, None)
            else:
                _PRIMITIVE_TRANSFORM_CACHE[63] = previous

        np.testing.assert_allclose(
            result.projected_primitive_lattice,
            transform @ proposed_conventional,
            atol=1.0e-10,
            rtol=1.0e-12,
        )
        audit = result.audit
        self.assertEqual(
            audit.schema,
            "wq_wyckoff_tangent_step_audit_v2",
        )
        self.assertEqual(
            audit.lattice_audit.projection_method,
            "global_chart_retraction_v1",
        )
        self.assertAlmostEqual(
            audit.lattice_audit.chart_update_norm,
            math.log(20.0),
            places=10,
        )
        self.assertLess(
            np.max(np.abs(result.projected_primitive_lattice)),
            100.0,
        )
        self.assertLessEqual(
            audit.primitive_transform_consistency_max_abs_error,
            1.0e-15,
        )
        self.assertLessEqual(
            audit.primitive_lattice_consistency_relative_error,
            1.0e-15,
        )

    def test_synthetic_two_and_three_dimensional_orbits(self) -> None:
        cases = (
            (
                2,
                (0.2, 0.3),
                np.asarray([[0.07, -0.04, 0.23]]),
                (0.27, 0.26),
                0.23,
            ),
            (
                3,
                (0.2, 0.3, 0.4),
                np.asarray([[0.07, -0.04, 0.09]]),
                (0.27, 0.26, 0.49),
                0.0,
            ),
        )
        for dimension, initial, update, expected, normal_norm in cases:
            with self.subTest(dimension=dimension):
                catalog = _CartesianCatalog(dimension)
                state = _state(
                    chart_dimension=dimension,
                    free_coordinate=initial,
                )
                expanded = expand_state(
                    state,
                    catalog,
                    redetect_space_group=False,
                )
                proposal = (
                    expanded.fractional_coordinates + update
                ) % 1.0
                result = WyckoffTangentProjector(
                    catalog
                ).project_parent_step(
                    state=state,
                    proposed_fractional_coordinates=proposal,
                    proposed_primitive_lattice=expanded.primitive_lattice,
                )
                np.testing.assert_allclose(
                    result.state.orbits[0].free_coordinate,
                    expected,
                    atol=2.0e-8,
                    rtol=0.0,
                )
                orbit = result.audit.orbit_audits[0]
                self.assertEqual(orbit.chart_dimension, dimension)
                self.assertEqual(orbit.jacobian_rank, dimension)
                self.assertAlmostEqual(
                    orbit.normal_residual_norm,
                    normal_norm,
                    places=7,
                )

    def test_parent_step_uses_minimum_image_and_removes_normal_update(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.95,))
        catalog = _LineCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        proposed_coordinates = expanded.fractional_coordinates.copy()
        proposed_coordinates[0] = np.asarray([0.05, 0.45, 0.20])
        proposed_lattice = expanded.primitive_lattice + np.asarray(
            [[0.01, 0.02, 0.00], [0.00, 0.01, 0.01], [0.00, 0.00, 0.02]]
        )

        result = WyckoffTangentProjector(catalog).project_parent_step(
            state=state,
            proposed_fractional_coordinates=proposed_coordinates,
            proposed_primitive_lattice=proposed_lattice,
        )

        self.assertAlmostEqual(
            result.state.orbits[0].free_coordinate[0],
            0.05,
            places=7,
        )
        np.testing.assert_allclose(
            result.projected_fractional_coordinates[0],
            np.asarray([0.05, 0.25, 0.5]),
            atol=2.0e-8,
            rtol=0.0,
        )
        orbit = result.audit.orbit_audits[0]
        self.assertEqual(orbit.chart_dimension, 1)
        self.assertEqual(orbit.jacobian_rank, 1)
        self.assertAlmostEqual(orbit.input_update_norm, math.sqrt(0.14), places=7)
        self.assertGreater(orbit.normal_residual_norm, 0.35)
        self.assertTrue(result.audit.discrete_fields_unchanged)
        self.assertTrue(result.audit.orbit_chart_roundtrip_checked)
        self.assertLessEqual(
            result.audit.maximum_orbit_chart_roundtrip_max_abs_error,
            1.0e-12,
        )
        self.assertEqual(
            result.audit.topology_hash_before,
            result.audit.topology_hash_after,
        )
        self.assertFalse(result.audit.retry_or_replacement_used)
        self.assertFalse(result.audit.topology_revision_used)
        json.dumps(result.audit.to_dict(), allow_nan=False)

    def test_fixed_site_drops_all_parent_coordinate_motion(self) -> None:
        state = _state(chart_dimension=0, free_coordinate=())
        catalog = _FixedCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        proposed_coordinates = np.asarray([[0.1, 0.2, 0.3]])

        result = WyckoffTangentProjector(catalog).project_parent_step(
            state=state,
            proposed_fractional_coordinates=proposed_coordinates,
            proposed_primitive_lattice=expanded.primitive_lattice,
        )

        self.assertEqual(result.state.orbits[0].free_coordinate, ())
        np.testing.assert_allclose(
            result.projected_fractional_coordinates,
            np.zeros((1, 3)),
            atol=0.0,
            rtol=0.0,
        )
        orbit = result.audit.orbit_audits[0]
        self.assertEqual(orbit.chart_dimension, 0)
        self.assertEqual(orbit.jacobian_rank, 0)
        self.assertEqual(orbit.fixed_site_drift, 0.0)
        self.assertAlmostEqual(
            orbit.normal_residual_norm,
            math.sqrt(0.14),
            places=12,
        )

    def test_rank_deficient_orbit_fails_before_projection(self) -> None:
        state = _state(chart_dimension=2, free_coordinate=(0.2, 0.3))
        catalog = _RankDeficientCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        proposed_coordinates = (
            expanded.fractional_coordinates
            + np.asarray([[0.1, 0.0, 0.0]])
        ) % 1.0
        with self.assertRaisesRegex(
            TangentProjectionError,
            r"Jacobian rank 1 != 2",
        ):
            WyckoffTangentProjector(catalog).project_parent_step(
                state=state,
                proposed_fractional_coordinates=proposed_coordinates,
                proposed_primitive_lattice=expanded.primitive_lattice,
            )

    def test_numerically_ill_conditioned_orbit_fails_closed(self) -> None:
        state = _state(chart_dimension=2, free_coordinate=(0.2, 0.3))
        catalog = _IllConditionedCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        with self.assertRaisesRegex(
            TangentProjectionError,
            r"Jacobian rank 1 != 2",
        ):
            WyckoffTangentProjector(catalog).project_parent_step(
                state=state,
                proposed_fractional_coordinates=(
                    expanded.fractional_coordinates
                    + np.asarray([[0.1, 0.0, 0.0]])
                )
                % 1.0,
                proposed_primitive_lattice=expanded.primitive_lattice,
            )

    def test_inputs_are_immutable_and_outputs_are_read_only(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.2,))
        catalog = _LineCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        proposed_coordinates = (
            expanded.fractional_coordinates
            + np.asarray([[0.03, 0.0, 0.0]])
        ) % 1.0
        proposed_lattice = expanded.primitive_lattice + np.eye(3) * 0.01
        state_snapshot = copy.deepcopy(state.to_dict())
        coordinates_snapshot = proposed_coordinates.copy()
        lattice_snapshot = proposed_lattice.copy()

        result = WyckoffTangentProjector(catalog).project_parent_step(
            state=state,
            proposed_fractional_coordinates=proposed_coordinates,
            proposed_primitive_lattice=proposed_lattice,
        )

        self.assertEqual(state.to_dict(), state_snapshot)
        np.testing.assert_array_equal(
            proposed_coordinates,
            coordinates_snapshot,
        )
        np.testing.assert_array_equal(proposed_lattice, lattice_snapshot)
        self.assertEqual(result.state.attempt_id, state.attempt_id)
        self.assertEqual(result.state.timestep, state.timestep)
        self.assertEqual(result.state.orbits[0].species, 14)
        self.assertEqual(result.state.orbits[0].multiplicity, 1)
        with self.assertRaises(ValueError):
            result.projected_fractional_coordinates[0, 0] = 0.0
        with self.assertRaises(ValueError):
            result.projected_primitive_lattice[0, 0] = 0.0

    def test_invalid_parent_coordinate_contract_fails_closed(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.2,))
        catalog = _LineCatalog()
        expanded = expand_state(state, catalog, redetect_space_group=False)
        with self.assertRaises(ValueError):
            WyckoffTangentProjector(catalog).project_parent_step(
                state=state,
                proposed_fractional_coordinates=np.asarray([[1.1, 0.25, 0.5]]),
                proposed_primitive_lattice=expanded.primitive_lattice,
            )
        with self.assertRaises(FloatingPointError):
            WyckoffTangentProjector(catalog).project_parent_step(
                state=state,
                proposed_fractional_coordinates=np.asarray(
                    [[np.nan, 0.25, 0.5]]
                ),
                proposed_primitive_lattice=expanded.primitive_lattice,
            )

    def test_orbit_permutation_does_not_change_projected_semantics(self) -> None:
        catalog = _TwoLineCatalog()
        lattice_chart = LatticeChartCodec.encode_matrix(
            _triclinic_lattice(),
            "triclinic",
        )
        first_orbits = (
            OrbitState("orbit-a", 0, 14, 1, 1, (0.2,)),
            OrbitState("orbit-b", 1, 8, 1, 1, (0.4,)),
        )
        first_state = StratifiedState(
            space_group=1,
            lattice_system="triclinic",
            lattice_chart=lattice_chart,
            orbits=first_orbits,
        )
        second_state = dataclasses.replace(
            first_state,
            orbits=tuple(reversed(first_orbits)),
        )

        def projected_by_id(state: StratifiedState) -> dict[str, float]:
            expanded = expand_state(
                state,
                catalog,
                redetect_space_group=False,
            )
            proposal = expanded.fractional_coordinates.copy()
            for orbit_index, orbit in enumerate(state.orbits):
                axis = 0 if orbit.wyckoff_type == 0 else 1
                proposal[orbit_index, axis] = (
                    proposal[orbit_index, axis] + 0.07
                ) % 1.0
            result = WyckoffTangentProjector(
                catalog
            ).project_parent_step(
                state=state,
                proposed_fractional_coordinates=proposal,
                proposed_primitive_lattice=expanded.primitive_lattice,
            )
            return {
                orbit.orbit_id: orbit.free_coordinate[0]
                for orbit in result.state.orbits
            }

        first = projected_by_id(first_state)
        second = projected_by_id(second_state)
        self.assertEqual(first.keys(), second.keys())
        for orbit_id in first:
            self.assertAlmostEqual(first[orbit_id], second[orbit_id], places=12)


class ManifoldNoiseProjectorTests(unittest.TestCase):
    def test_paired_parent_noise_is_deterministically_projected_to_tangent(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.3,))
        catalog = _LineCatalog()
        parent_coordinate_noise = np.asarray([[2.0, 3.0, 4.0]])
        parent_lattice_noise = np.asarray(
            [
                [1.0, 0.2, -0.3],
                [0.4, 2.0, 0.5],
                [-0.6, 0.7, 3.0],
            ]
        )
        coordinate_snapshot = parent_coordinate_noise.copy()
        lattice_snapshot = parent_lattice_noise.copy()

        projector = ManifoldNoiseProjector(catalog)
        first = projector.project_parent_noise(
            state=state,
            parent_coordinate_noise=parent_coordinate_noise,
            parent_primitive_lattice_noise=parent_lattice_noise,
        )
        second = projector.project_parent_noise(
            state=state,
            parent_coordinate_noise=parent_coordinate_noise,
            parent_primitive_lattice_noise=parent_lattice_noise,
        )

        np.testing.assert_array_equal(
            parent_coordinate_noise,
            coordinate_snapshot,
        )
        np.testing.assert_array_equal(parent_lattice_noise, lattice_snapshot)
        np.testing.assert_array_equal(
            first.coordinate_noise,
            second.coordinate_noise,
        )
        np.testing.assert_array_equal(
            first.primitive_lattice_noise,
            second.primitive_lattice_noise,
        )
        np.testing.assert_allclose(
            first.coordinate_noise,
            np.asarray([[2.0 * math.sqrt(3.0), 0.0, 0.0]]),
            atol=1.0e-10,
            rtol=0.0,
        )
        orbit = first.audit.coordinate_subspaces[0]
        self.assertEqual(orbit.ambient_dimension, 3)
        self.assertEqual(orbit.tangent_dimension, 1)
        self.assertAlmostEqual(orbit.energy_scale, math.sqrt(3.0))
        self.assertLessEqual(orbit.orthonormality_max_abs_error, 1.0e-12)
        self.assertLessEqual(orbit.tangent_residual_norm, 1.0e-12)
        self.assertEqual(
            first.audit.distribution_claim,
            "parent_schedule_manifold_restricted_not_parent_iid",
        )
        self.assertTrue(first.audit.paired_parent_noise_transform)
        self.assertFalse(first.audit.parent_schedule_changed)
        with self.assertRaises(ValueError):
            first.coordinate_noise[0, 0] = 0.0
        with self.assertRaises(ValueError):
            first.primitive_lattice_noise[0, 0] = 0.0

    def test_fixed_site_receives_exactly_zero_coordinate_noise(self) -> None:
        state = _state(chart_dimension=0, free_coordinate=())
        catalog = _FixedCatalog()
        result = ManifoldNoiseProjector(catalog).project_parent_noise(
            state=state,
            parent_coordinate_noise=np.asarray([[1.0, -2.0, 3.0]]),
            parent_primitive_lattice_noise=np.eye(3),
        )
        np.testing.assert_array_equal(
            result.coordinate_noise,
            np.zeros((1, 3)),
        )
        self.assertTrue(result.audit.fixed_site_noise_zero)
        self.assertEqual(
            result.audit.coordinate_subspaces[0].tangent_dimension,
            0,
        )

    def test_cubic_lattice_noise_has_only_one_symmetric_degree_of_freedom(self) -> None:
        state = dataclasses.replace(
            _state(chart_dimension=1, free_coordinate=(0.3,)),
            lattice_system="cubic",
            lattice_chart=LatticeChartCodec.encode_matrix(
                np.diag([4.0, 4.0, 4.0]),
                "cubic",
            ),
        )
        base = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        result = ManifoldNoiseProjector(_LineCatalog()).project_parent_noise(
            state=state,
            parent_coordinate_noise=np.asarray([[0.1, 0.2, 0.3]]),
            parent_primitive_lattice_noise=base,
        )
        lattice_noise = result.primitive_lattice_noise
        np.testing.assert_allclose(
            lattice_noise - np.diag(np.diag(lattice_noise)),
            np.zeros((3, 3)),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertAlmostEqual(lattice_noise[0, 0], lattice_noise[1, 1])
        self.assertAlmostEqual(lattice_noise[1, 1], lattice_noise[2, 2])
        lattice = result.audit.lattice_subspace
        self.assertEqual(lattice.ambient_dimension, 9)
        self.assertEqual(lattice.tangent_dimension, 1)
        self.assertEqual(lattice.jacobian_rank, 1)
        self.assertAlmostEqual(lattice.energy_scale, 3.0)
        self.assertLessEqual(lattice.tangent_residual_norm, 1.0e-12)

    def test_forward_state_reuses_parent_schedule_in_wq_charts(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.3,))
        catalog = _LineCatalog()
        schedules = build_numpy_parent_schedules()
        cells = build_bridge_cells(base_seed=2026072601)

        results = [
            forward_noise_manifold_numpy(
                state=state,
                catalog=catalog,
                schedules=schedules,
                cell=cell,
            )
            for cell in cells
            if cell.panel_index == 0
        ]

        self.assertEqual(
            [result.audit.timestep for result in results],
            [100, 200, 400, 800],
        )
        for result in results:
            timestep = result.audit.timestep
            self.assertEqual(
                result.audit.alpha_bar,
                float(schedules.alphas_cumprod[timestep]),
            )
            self.assertEqual(
                result.audit.coordinate_sigma,
                float(schedules.coordinate_sigmas[timestep]),
            )
            self.assertEqual(result.state.timestep, timestep / 1000)
            self.assertTrue(result.audit.discrete_fields_unchanged)
            self.assertEqual(
                result.audit.condition_topology_hash,
                result.audit.noised_topology_hash,
            )
            self.assertFalse(result.audit.parent_schedule_changed)
            self.assertFalse(
                result.audit.clean_condition_used_as_reverse_state
            )
            self.assertFalse(result.audit.retry_or_replacement_used)
            self.assertLessEqual(
                result.audit.coordinate_chart_reconstruction_max_abs_error,
                1.0e-12,
            )
            self.assertLessEqual(
                result.audit.lattice_chart_reconstruction_max_abs_error,
                1.0e-12,
            )
            self.assertGreater(result.audit.primitive_volume, 0.0)
            np.testing.assert_allclose(
                result.fractional_coordinates[:, 1:],
                np.asarray([[0.25, 0.5]]),
                atol=1.0e-12,
                rtol=0.0,
            )
        repeated = forward_noise_manifold_numpy(
            state=state,
            catalog=catalog,
            schedules=schedules,
            cell=cells[0],
        )
        np.testing.assert_array_equal(
            results[0].fractional_coordinates,
            repeated.fractional_coordinates,
        )
        np.testing.assert_array_equal(
            results[0].primitive_lattice,
            repeated.primitive_lattice,
        )

    def test_rank_deficient_noise_chart_fails_closed(self) -> None:
        state = _state(chart_dimension=2, free_coordinate=(0.2, 0.3))
        with self.assertRaisesRegex(
            TangentProjectionError,
            r"Jacobian rank 1 != 2",
        ):
            ManifoldNoiseProjector(
                _RankDeficientCatalog()
            ).project_parent_noise(
                state=state,
                parent_coordinate_noise=np.asarray([[1.0, 2.0, 3.0]]),
                parent_primitive_lattice_noise=np.eye(3),
            )

    def test_reverse_identity_binds_forward_state_to_parent_batch(self) -> None:
        state = _state(chart_dimension=1, free_coordinate=(0.3,))
        catalog = _LineCatalog()
        forward = forward_noise_manifold_numpy(
            state=state,
            catalog=catalog,
            schedules=build_numpy_parent_schedules(),
            cell=build_bridge_cells(base_seed=2026072601)[0],
        )
        audit = validate_manifold_reverse_identity(
            forward_state=forward,
            projector=WyckoffTangentProjector(catalog),
            batch_num_atoms=np.asarray([1]),
            batch_atomic_numbers=np.asarray([14]),
            batch_fractional_coordinates=np.asarray([[0.3, 0.25, 0.5]]),
            batch_assignments=np.asarray([0]),
            batch_clean_primitive_lattice=_triclinic_lattice(),
        )
        self.assertEqual(audit.atom_count, 1)
        self.assertTrue(audit.batch_species_and_order_match)
        self.assertTrue(audit.single_graph_batch_match)
        self.assertFalse(audit.clean_batch_geometry_used_as_reverse_state)
        self.assertLessEqual(
            audit.primitive_coordinate_max_abs_error,
            1.0e-12,
        )
        self.assertLessEqual(
            audit.primitive_lattice_max_abs_error,
            1.0e-12,
        )
        with self.assertRaisesRegex(
            TangentProjectionError,
            "graph/species identity",
        ):
            validate_manifold_reverse_identity(
                forward_state=forward,
                projector=WyckoffTangentProjector(catalog),
                batch_num_atoms=np.asarray([1]),
                batch_atomic_numbers=np.asarray([8]),
                batch_fractional_coordinates=np.asarray(
                    [[0.3, 0.25, 0.5]]
                ),
                batch_assignments=np.asarray([0]),
                batch_clean_primitive_lattice=_triclinic_lattice(),
            )


class TangentBridgeStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TANGENT_BRIDGE_SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_module_has_no_eager_torch_or_heavy_crystal_import(self) -> None:
        forbidden = {"torch", "pymatgen", "spglib", "pyxtal"}
        imported: set[str] = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertTrue(forbidden.isdisjoint(imported))

    def test_reverse_path_reuses_parent_decoder_and_schedule_formula(self) -> None:
        source = inspect.getsource(run_parent_reverse_on_wyckoff_manifold)
        self.assertEqual(source.count("model.decoder("), 2)
        self.assertIn("model.time_embedding", source)
        self.assertIn("model.sigma_scheduler.sigmas", source)
        self.assertIn("model.beta_scheduler.alphas_cumprod", source)
        self.assertIn("respaced_timesteps", source)
        self.assertEqual(source.count("project_candidate("), 3)
        self.assertIn("calls != 2 * reverse_steps", source)
        self.assertIn("validate_manifold_reverse_identity", source)
        self.assertIn('"input_identity_verified": True', source)
        self.assertIn("lattice_frame.to_wq", source)
        self.assertIn("lattice_frame.to_parent", source)

    def test_module_contains_no_training_mlip_api_or_retry_path(self) -> None:
        lowered = self.source.lower()
        for forbidden in (
            "chgnet",
            "mattersim",
            "mprester",
            "requests.",
            ".backward(",
            "optimizer",
            "retry(",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertFalse(
            any(isinstance(node, ast.While) for node in ast.walk(self.tree))
        )


if __name__ == "__main__":
    unittest.main()

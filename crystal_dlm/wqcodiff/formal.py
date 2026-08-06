"""Numerical audits that gate use of stratified/trans-dimensional claims."""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable, Sequence

import numpy as np

from .kernel import TopologyEventKernel, TransitionError
from .state import StratifiedState


@dataclasses.dataclass(frozen=True, slots=True)
class FormalAuditResult:
    transitions: int
    illegal_transitions: int
    normalization_error_max: float
    target_stratum_failures: int
    initial_dimension: int
    final_dimension: int

    @property
    def passed(self) -> bool:
        return (
            self.illegal_transitions == 0
            and self.target_stratum_failures == 0
            and self.normalization_error_max < 1.0e-6
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PyXtalCatalogAuditResult:
    space_groups: int
    wyckoff_positions: int
    free_roundtrip_error_max: float
    affine_decode_error_max: float
    affine_encode_error_max: float
    expansion_alignment_error_max: float
    projector_error_max: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.space_groups == 230
            # Exact for PyXtal 1.1.4.  This catches the easy-to-miss SG 47
            # upper-case A position, which must not alias lower-case a.
            and self.wyckoff_positions == 1731
            and not self.failures
            and self.free_roundtrip_error_max < 1.0e-6
            and self.affine_decode_error_max < 1.0e-6
            and self.affine_encode_error_max < 1.0e-6
            and self.expansion_alignment_error_max < 1.0e-12
            and self.projector_error_max < 1.0e-6
        )


def chart_roundtrip_error(
    coordinates: Sequence[float],
    decode: Callable[[Sequence[float]], Sequence[float]],
    encode: Callable[[Sequence[float]], Sequence[float]],
) -> float:
    reconstructed = tuple(float(v) for v in encode(decode(coordinates)))
    if len(reconstructed) != len(coordinates):
        return math.inf
    return max((abs(((a - b + 0.5) % 1.0) - 0.5) for a, b in zip(coordinates, reconstructed)), default=0.0)


def regularized_projector_error(
    jacobians: Sequence[Sequence[Sequence[float]]],
    *,
    ridge: float = 1.0e-8,
) -> float:
    """Measure recovery error of the registered tangent pseudoinverse.

    The input contains one ``3 x d`` Jacobian per expanded atom.  The returned
    infinity norm is the worst error when projecting every free-coordinate
    basis vector to atoms and back with the exact float64 projector used at
    sampling time.
    """

    values = np.asarray(jacobians, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 3:
        return math.inf
    dimension = int(values.shape[2])
    if dimension == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return math.inf
    design = values.reshape(-1, dimension)
    if np.linalg.matrix_rank(design, tol=1.0e-12) != dimension:
        return math.inf
    gram = design.T @ design
    recovered = np.linalg.solve(
        gram + float(ridge) * np.eye(dimension, dtype=np.float64),
        gram,
    )
    return float(np.max(np.abs(recovered - np.eye(dimension, dtype=np.float64))))


def audit_pyxtal_chart_catalog() -> PyXtalCatalogAuditResult:
    """Exhaustively audit all 230 SGs in the pinned PyXtal chart convention."""

    from .charts import PyXtalChartCatalog, periodic_cartesian_distance

    catalog = PyXtalChartCatalog()
    lattice = np.eye(3, dtype=np.float64)
    position_count = 0
    free_error = affine_decode_error = affine_encode_error = 0.0
    expansion_alignment_error = projector_error = 0.0
    failures: list[str] = []
    groups_seen = 0
    for space_group in range(1, 231):
        try:
            types = catalog.types(space_group)
            if not types:
                raise ValueError("empty Wyckoff vocabulary")
            catalog.hall_number(space_group)
            groups_seen += 1
        except Exception as exc:
            failures.append(f"SG{space_group}:catalog:{type(exc).__name__}:{exc}")
            continue
        for wyckoff_type in types:
            position_count += 1
            label = f"SG{space_group}:W{wyckoff_type}"
            try:
                spec = catalog.get(space_group, wyckoff_type)
                free = tuple(
                    0.2
                    + (
                        0.137 * (axis + 1)
                        + 0.001 * space_group
                        + 0.003 * wyckoff_type
                    )
                    % 0.6
                    for axis in range(spec.dimension)
                )
                exact = catalog.decode_generator(space_group, wyckoff_type, free)
                recovered = catalog.encode_free(space_group, wyckoff_type, exact)
                free_error = max(
                    free_error,
                    max(
                        (
                            abs(((left - right + 0.5) % 1.0) - 0.5)
                            for left, right in zip(free, recovered)
                        ),
                        default=0.0,
                    ),
                )
                chart = catalog.affine_chart(space_group, wyckoff_type, lattice)
                affine = chart.decode(free)
                affine_decode_error = max(
                    affine_decode_error,
                    periodic_cartesian_distance(exact, affine, lattice),
                )
                affine_recovered = chart.encode(exact)
                affine_encode_error = max(
                    affine_encode_error,
                    max(
                        (
                            abs(((left - right + 0.5) % 1.0) - 0.5)
                            for left, right in zip(free, affine_recovered)
                        ),
                        default=0.0,
                    ),
                )
                expanded = catalog.expand(space_group, wyckoff_type, free)
                if len(expanded) != spec.multiplicity:
                    raise ValueError("expanded multiplicity mismatch")
                expanded_with_jacobians = catalog.expand_with_jacobians(
                    space_group, wyckoff_type, free
                )
                aligned = tuple(point for point, _ in expanded_with_jacobians)
                if len(aligned) != len(expanded):
                    raise ValueError("exact Jacobian expansion multiplicity mismatch")
                point_errors = tuple(
                    periodic_cartesian_distance(left, right, lattice)
                    for left, right in zip(expanded, aligned)
                )
                alignment_error = max(point_errors, default=0.0)
                expansion_alignment_error = max(
                    expansion_alignment_error, alignment_error
                )
                # Both paths are sorted by the same 12-decimal fractional key.
                # Test periodic geometry at a much tighter tolerance than the
                # paper gate instead of requiring bitwise-identical floats.
                if alignment_error >= 1.0e-12:
                    raise ValueError(
                        "exact Jacobian expansion ordering/geometry mismatch: "
                        f"{alignment_error}"
                    )
                projector_error = max(
                    projector_error,
                    regularized_projector_error(
                        [jacobian for _, jacobian in expanded_with_jacobians]
                    ),
                )
            except Exception as exc:
                failures.append(f"{label}:{type(exc).__name__}:{exc}")
    return PyXtalCatalogAuditResult(
        space_groups=groups_seen,
        wyckoff_positions=position_count,
        free_roundtrip_error_max=free_error,
        affine_decode_error_max=affine_decode_error,
        affine_encode_error_max=affine_encode_error,
        expansion_alignment_error_max=expansion_alignment_error,
        projector_error_max=projector_error,
        failures=tuple(failures),
    )


def permutation_posterior_difference(
    state: StratifiedState,
    posterior: Callable[[StratifiedState], Sequence[float]],
    *,
    seed: int = 0,
    trials: int = 32,
) -> float:
    baseline = tuple(float(value) for value in posterior(state))
    maximum = 0.0
    rng = random.Random(seed)
    for _ in range(trials):
        candidate = tuple(float(value) for value in posterior(state.permuted(rng)))
        if len(candidate) != len(baseline):
            return math.inf
        maximum = max(maximum, max((abs(a - b) for a, b in zip(baseline, candidate)), default=0.0))
    return maximum


def run_synthetic_transition_audit(
    kernel: TopologyEventKernel,
    initial_state: StratifiedState,
    *,
    transitions: int = 1_000_000,
    seed: int = 0,
) -> FormalAuditResult:
    if transitions <= 0:
        raise ValueError("transitions must be positive")
    rng = random.Random(seed)
    state = initial_state
    initial_dimension = state.continuous_dimension
    illegal = 0
    target_failures = 0
    normalization_error = 0.0
    for _ in range(transitions):
        legal = kernel.legal_events(state)
        # A deterministic, broad logit pattern exercises normalization without
        # introducing platform-specific random math.
        logits = {event: ((index * 17) % 31 - 15) / 7.0 for index, event in enumerate(legal)}
        weighted = kernel.probabilities(state, logits)
        normalization_error = max(
            normalization_error,
            abs(math.fsum(probability for _, probability in weighted) - 1.0),
        )
        event = kernel.sample(state, logits, rng)
        if event not in legal:
            illegal += 1
            continue
        try:
            state = kernel.apply(state, event, rng)
        except TransitionError:
            target_failures += 1
            # Bridge failures are terminal in real sampling.  For an audit we
            # reset to continue measuring the entire requested transition set.
            state = initial_state
    return FormalAuditResult(
        transitions=transitions,
        illegal_transitions=illegal,
        normalization_error_max=normalization_error,
        target_stratum_failures=target_failures,
        initial_dimension=initial_dimension,
        final_dimension=state.continuous_dimension,
    )

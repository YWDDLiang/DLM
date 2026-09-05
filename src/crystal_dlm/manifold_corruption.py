"""Coherent crystal corruptions on ``SPD(3) x T^(3N)``.

The module is deliberately model- and potential-free.  It produces
request-keyed geometric proposals, sends every proposal through the native
dynamic ``7 + 4N`` codec, and delegates physical certification to a callback.
This keeps MLIP use outside the reusable data plumbing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from crystal_dlm.dynamic_crystal import (
    arrays_to_dynamic_answer,
    parse_dynamic_answer,
)


Array = np.ndarray


def _finite_array(value: Any, *, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def _symmetric(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)


def _eigh_spd(matrix: Array, *, name: str) -> tuple[Array, Array]:
    value = _finite_array(matrix, shape=(3, 3), name=name)
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(value))
    if float(eigenvalues.min()) <= 1.0e-10:
        raise ValueError(f"{name} must be positive definite")
    return eigenvalues, eigenvectors


def spd_matrix_power(matrix: Array, power: float) -> Array:
    """Return a symmetric real power of a 3x3 SPD matrix."""

    eigenvalues, eigenvectors = _eigh_spd(matrix, name="SPD matrix")
    powered = np.power(eigenvalues, float(power))
    return _symmetric((eigenvectors * powered) @ eigenvectors.T)


def symmetric_matrix_exp(matrix: Array) -> Array:
    """Exponentiate a finite symmetric 3x3 matrix."""

    value = _finite_array(matrix, shape=(3, 3), name="symmetric matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(value))
    return _symmetric((eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T)


def symmetric_matrix_log(matrix: Array) -> Array:
    """Return the principal logarithm of a 3x3 SPD matrix."""

    eigenvalues, eigenvectors = _eigh_spd(matrix, name="SPD matrix")
    return _symmetric((eigenvectors * np.log(eigenvalues)) @ eigenvectors.T)


def lattice_matrix_from_parameters(
    lengths: Sequence[float], angles_deg: Sequence[float]
) -> Array:
    """Construct the repository's canonical row-vector lattice matrix."""

    if len(lengths) != 3 or len(angles_deg) != 3:
        raise ValueError("lattice parameters require three lengths and three angles")
    a, b, c = (float(value) for value in lengths)
    alpha, beta, gamma = (math.radians(float(value)) for value in angles_deg)
    if min(a, b, c) <= 0.0:
        raise ValueError("lattice lengths must be positive")
    if not all(0.0 < value < math.pi for value in (alpha, beta, gamma)):
        raise ValueError("lattice angles must lie in (0, 180) degrees")
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) <= 1.0e-8:
        raise ValueError("lattice gamma produces a singular canonical cell")
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 1.0e-10:
        raise ValueError("lattice parameters do not define a positive volume")
    return np.asarray(
        (
            (a, 0.0, 0.0),
            (b * math.cos(gamma), b * sin_gamma, 0.0),
            (cx, cy, math.sqrt(cz_squared)),
        ),
        dtype=float,
    )


def lattice_parameters_from_matrix(lattice: Array) -> tuple[list[float], list[float]]:
    """Convert a finite row-vector lattice to lengths and inter-vector angles."""

    value = _finite_array(lattice, shape=(3, 3), name="lattice")
    lengths = np.linalg.norm(value, axis=1)
    if float(lengths.min()) <= 1.0e-10 or abs(float(np.linalg.det(value))) <= 1.0e-10:
        raise ValueError("lattice must have positive non-zero vector lengths and volume")

    def angle(left: int, right: int) -> float:
        cosine = float(np.dot(value[left], value[right]) / (lengths[left] * lengths[right]))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    # alpha=(b,c), beta=(a,c), gamma=(a,b)
    angles = [angle(1, 2), angle(0, 2), angle(0, 1)]
    return [float(item) for item in lengths], angles


def canonical_lattice_from_metric(metric: Array) -> Array:
    """Return the unique lower-triangular row lattice ``L`` with ``G=L L^T``."""

    _eigh_spd(metric, name="metric")
    return np.linalg.cholesky(_symmetric(metric))


def torus_delta(target: Array, source: Array) -> Array:
    """Shortest component-wise fractional displacement from source to target."""

    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    return delta - np.round(delta)


def minimum_image_cartesian_retraction(
    target_fractional: Array,
    source_fractional: Array,
    lattice: Array,
    *,
    image_radius: int = 2,
) -> Array:
    """Return exact searched PBC vectors from source sites to target sites."""

    target = np.asarray(target_fractional, dtype=float)
    source = np.asarray(source_fractional, dtype=float)
    if target.shape != source.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target/source fractional coordinates must share shape [N, 3]")
    cell = _finite_array(lattice, shape=(3, 3), name="lattice")
    radius = int(image_radius)
    if radius < 1:
        raise ValueError("image_radius must be positive")
    shifts = np.asarray(
        list(itertools.product(range(-radius, radius + 1), repeat=3)),
        dtype=float,
    )
    centered = torus_delta(target, source)
    candidates = (centered[:, None, :] + shifts[None, :, :]) @ cell
    squared = np.einsum("nij,nij->ni", candidates, candidates)
    selected = np.argmin(squared, axis=1)
    return candidates[np.arange(len(source)), selected]


@dataclass(frozen=True)
class CrystalGeometry:
    """A crystal in the row-vector lattice convention used by this repository."""

    lattice: Array
    frac_coords: Array
    species: tuple[str, ...]

    def __post_init__(self) -> None:
        lattice = _finite_array(self.lattice, shape=(3, 3), name="lattice")
        coordinates = np.asarray(self.frac_coords, dtype=float)
        if coordinates.shape != (len(self.species), 3) or not np.isfinite(coordinates).all():
            raise ValueError("frac_coords must be finite with shape [N, 3]")
        if not self.species or any(not str(symbol) for symbol in self.species):
            raise ValueError("species must contain at least one non-empty symbol")
        metric = lattice @ lattice.T
        _eigh_spd(metric, name="lattice metric")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "frac_coords", np.mod(coordinates, 1.0))
        object.__setattr__(self, "species", tuple(str(symbol) for symbol in self.species))

    @property
    def metric(self) -> Array:
        return _symmetric(self.lattice @ self.lattice.T)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CrystalGeometry":
        species = tuple(str(symbol) for symbol in value["species"])
        coordinates = np.asarray(value["frac_coords"], dtype=float)
        if "lattice" in value:
            lattice = np.asarray(value["lattice"], dtype=float)
        else:
            lattice = lattice_matrix_from_parameters(value["lengths"], value["angles"])
        return cls(lattice=lattice, frac_coords=coordinates, species=species)


@dataclass(frozen=True)
class CorruptionConfig:
    """Bounded proposal and certificate settings for coherent corruption."""

    max_proposals: int = 4
    lattice_log_std: float = 0.055
    coordinate_cartesian_std_A: float = 0.18
    max_logmetric_frobenius: float = 0.24
    max_atom_displacement_A: float = 0.60
    max_delta_energy: float = 2.0
    require_lattice_token_change: bool = True
    require_coordinate_token_change: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_proposals) <= 4:
            raise ValueError("max_proposals must be in 1..4")
        if min(
            float(self.lattice_log_std),
            float(self.coordinate_cartesian_std_A),
            float(self.max_logmetric_frobenius),
            float(self.max_atom_displacement_A),
            float(self.max_delta_energy),
        ) <= 0.0:
            raise ValueError("corruption scales and energy bound must be positive")


@dataclass(frozen=True)
class CorruptionCertificate:
    """Compact physical certificate supplied by an external train-only provider."""

    post_quantization_valid: bool
    delta_energy: float
    coordinate_force_dot_clean_retraction: float | None
    lattice_descent_dot_spd_retraction: float | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CorruptionCertificate":
        def optional_float(name: str) -> float | None:
            raw = value.get(name)
            return None if raw is None else float(raw)

        return cls(
            post_quantization_valid=bool(value.get("post_quantization_valid", False)),
            delta_energy=float(value.get("delta_energy", math.nan)),
            coordinate_force_dot_clean_retraction=optional_float(
                "coordinate_force_dot_clean_retraction"
            ),
            lattice_descent_dot_spd_retraction=optional_float(
                "lattice_descent_dot_spd_retraction"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def accepts(self, proposal: "CorruptionProposal", config: CorruptionConfig) -> bool:
        if not self.post_quantization_valid:
            return False
        if not math.isfinite(self.delta_energy) or not (
            0.0 < self.delta_energy <= float(config.max_delta_energy)
        ):
            return False
        if proposal.coordinate_active:
            value = self.coordinate_force_dot_clean_retraction
            if value is None or not math.isfinite(value) or value <= 0.0:
                return False
        if proposal.lattice_active:
            value = self.lattice_descent_dot_spd_retraction
            if value is None or not math.isfinite(value) or value <= 0.0:
                return False
        return True


@dataclass(frozen=True)
class CorruptionProposal:
    """One continuous proposal after native-token quantization and decoding."""

    request_key: str
    proposal_index: int
    clean_body: str
    body: str
    clean_tokens: tuple[str, ...]
    tokens: tuple[str, ...]
    clean_geometry: CrystalGeometry
    geometry: CrystalGeometry
    sampled_logmetric_tangent: Array
    sampled_zero_com_cartesian_displacement: Array
    clean_coordinate_retraction_cartesian: Array
    clean_spd_retraction_tangent: Array
    lattice_changed_positions: tuple[int, ...]
    coordinate_changed_positions: tuple[int, ...]
    encoding_clipped: bool

    @property
    def lattice_active(self) -> bool:
        return bool(self.lattice_changed_positions)

    @property
    def coordinate_active(self) -> bool:
        return bool(self.coordinate_changed_positions)

    def has_required_changes(self, config: CorruptionConfig) -> bool:
        lattice_ok = self.lattice_active or not config.require_lattice_token_change
        coordinate_requested = len(self.clean_geometry.species) > 1
        coordinate_ok = (
            self.coordinate_active
            or not config.require_coordinate_token_change
            or not coordinate_requested
        )
        return bool(lattice_ok and coordinate_ok and not self.encoding_clipped)


@dataclass(frozen=True)
class CorruptionSelection:
    """First certified proposal, or a clean-CE fallback when none certifies."""

    clean_body: str
    clean_geometry: CrystalGeometry
    attempted_proposals: int
    proposal: CorruptionProposal | None
    certificate: CorruptionCertificate | None

    @property
    def fallback(self) -> bool:
        return self.proposal is None

    @property
    def source_body(self) -> str:
        return self.clean_body if self.proposal is None else self.proposal.body


CertificateCallback = Callable[
    [CorruptionProposal], CorruptionCertificate | Mapping[str, Any] | None
]


def canonical_request_key(value: Any) -> str:
    """Return a stable textual request identity for deterministic seeding."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def request_seed(*, seed: int, request_key: Any, proposal_index: int) -> int:
    """Derive a platform-stable RNG seed from request identity and proposal index."""

    key = canonical_request_key(request_key)
    payload = f"pmtr-v1\0{int(seed)}\0{key}\0{int(proposal_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _roundtrip_body(body: str) -> tuple[str, tuple[str, ...], CrystalGeometry]:
    parsed = parse_dynamic_answer(str(body), strict=True)
    tokens = tuple(str(token) for token in parsed["tokens"])
    canonical_body = "".join(tokens)
    if canonical_body != str(body):
        raise ValueError("dynamic body is not in canonical separator-free 7+4N form")
    return canonical_body, tokens, CrystalGeometry.from_mapping(parsed)


def _roundtrip_geometry(
    geometry: CrystalGeometry,
) -> tuple[str, tuple[str, ...], CrystalGeometry, bool]:
    lengths, angles = lattice_parameters_from_matrix(geometry.lattice)
    body, diagnostics = arrays_to_dynamic_answer(
        lengths=lengths,
        angles=angles,
        species=geometry.species,
        frac_coords=geometry.frac_coords,
        separator="",
    )
    canonical_body, tokens, decoded = _roundtrip_body(body)
    clipped = bool(
        diagnostics.length_clips
        or diagnostics.angle_clips
        or diagnostics.coord_clips
    )
    return canonical_body, tokens, decoded, clipped


def _bounded_symmetric_tangent(rng: np.random.Generator, config: CorruptionConfig) -> Array:
    raw = rng.normal(0.0, float(config.lattice_log_std), size=(3, 3))
    tangent = _symmetric(raw)
    norm = float(np.linalg.norm(tangent, ord="fro"))
    if norm > float(config.max_logmetric_frobenius):
        tangent *= float(config.max_logmetric_frobenius) / norm
    return tangent


def _zero_com_cartesian_displacement(
    rng: np.random.Generator,
    count: int,
    config: CorruptionConfig,
) -> Array:
    displacement = rng.normal(
        0.0, float(config.coordinate_cartesian_std_A), size=(int(count), 3)
    )
    displacement -= displacement.mean(axis=0, keepdims=True)
    maximum = float(np.linalg.norm(displacement, axis=1).max(initial=0.0))
    if maximum > float(config.max_atom_displacement_A):
        displacement *= float(config.max_atom_displacement_A) / maximum
    return displacement


def _affine_spd_retraction(source_metric: Array, target_metric: Array) -> Array:
    inverse_sqrt = spd_matrix_power(source_metric, -0.5)
    whitened = inverse_sqrt @ target_metric @ inverse_sqrt
    return symmetric_matrix_log(_symmetric(whitened))


def generate_corruption_proposal(
    clean: CrystalGeometry,
    *,
    clean_body: str,
    clean_tokens: Sequence[str],
    request_key: Any,
    proposal_index: int,
    seed: int,
    config: CorruptionConfig = CorruptionConfig(),
) -> CorruptionProposal:
    """Generate and quantize one coherent lattice-plus-coordinate proposal."""

    key = canonical_request_key(request_key)
    rng = np.random.default_rng(
        request_seed(seed=int(seed), request_key=key, proposal_index=int(proposal_index))
    )
    tangent = _bounded_symmetric_tangent(rng, config)
    clean_metric_sqrt = spd_matrix_power(clean.metric, 0.5)
    corrupted_metric = _symmetric(
        clean_metric_sqrt @ symmetric_matrix_exp(tangent) @ clean_metric_sqrt
    )
    corrupted_lattice = canonical_lattice_from_metric(corrupted_metric)

    displacement = _zero_com_cartesian_displacement(rng, len(clean.species), config)
    clean_cartesian = clean.frac_coords @ clean.lattice
    corrupted_cartesian = clean_cartesian + displacement
    corrupted_fractional = np.mod(
        corrupted_cartesian @ np.linalg.inv(corrupted_lattice), 1.0
    )
    continuous = CrystalGeometry(
        lattice=corrupted_lattice,
        frac_coords=corrupted_fractional,
        species=clean.species,
    )
    body, tokens, decoded, clipped = _roundtrip_geometry(continuous)
    if tuple(decoded.species) != tuple(clean.species):
        raise RuntimeError("native codec changed species/site order")
    if len(tokens) != len(clean_tokens):
        raise RuntimeError("native codec changed exact 7+4N length")
    changed = tuple(
        position
        for position, (before, after) in enumerate(
            zip(clean_tokens, tokens, strict=True)
        )
        if before != after
    )
    lattice_changed = tuple(position for position in changed if 1 <= position <= 6)
    coordinate_changed = tuple(
        position for position in changed if position >= 7 and (position - 7) % 4 != 0
    )
    clean_coordinate_retraction = minimum_image_cartesian_retraction(
        clean.frac_coords,
        decoded.frac_coords,
        decoded.lattice,
        image_radius=2,
    )
    clean_spd_retraction = _affine_spd_retraction(decoded.metric, clean.metric)
    return CorruptionProposal(
        request_key=key,
        proposal_index=int(proposal_index),
        clean_body=str(clean_body),
        body=body,
        clean_tokens=tuple(str(token) for token in clean_tokens),
        tokens=tokens,
        clean_geometry=clean,
        geometry=decoded,
        sampled_logmetric_tangent=tangent,
        sampled_zero_com_cartesian_displacement=displacement,
        clean_coordinate_retraction_cartesian=clean_coordinate_retraction,
        clean_spd_retraction_tangent=clean_spd_retraction,
        lattice_changed_positions=lattice_changed,
        coordinate_changed_positions=coordinate_changed,
        encoding_clipped=bool(clipped),
    )


def select_first_certified_corruption(
    clean: str | CrystalGeometry | Mapping[str, Any],
    *,
    request_key: Any,
    certify: CertificateCallback,
    seed: int,
    config: CorruptionConfig = CorruptionConfig(),
) -> CorruptionSelection:
    """Return the first certified proposal, never the lowest-energy proposal."""

    if isinstance(clean, str):
        clean_body, clean_tokens, clean_geometry = _roundtrip_body(clean)
    else:
        geometry = (
            clean
            if isinstance(clean, CrystalGeometry)
            else CrystalGeometry.from_mapping(clean)
        )
        clean_body, clean_tokens, clean_geometry, clipped = _roundtrip_geometry(geometry)
        if clipped:
            raise ValueError("clean geometry clips in the native 7+4N codec")

    attempted = 0
    for proposal_index in range(int(config.max_proposals)):
        attempted += 1
        proposal = generate_corruption_proposal(
            clean_geometry,
            clean_body=clean_body,
            clean_tokens=clean_tokens,
            request_key=request_key,
            proposal_index=proposal_index,
            seed=int(seed),
            config=config,
        )
        if not proposal.has_required_changes(config):
            continue
        raw_certificate = certify(proposal)
        if raw_certificate is None:
            continue
        certificate = (
            raw_certificate
            if isinstance(raw_certificate, CorruptionCertificate)
            else CorruptionCertificate.from_mapping(raw_certificate)
        )
        if certificate.accepts(proposal, config):
            return CorruptionSelection(
                clean_body=clean_body,
                clean_geometry=clean_geometry,
                attempted_proposals=attempted,
                proposal=proposal,
                certificate=certificate,
            )
    return CorruptionSelection(
        clean_body=clean_body,
        clean_geometry=clean_geometry,
        attempted_proposals=attempted,
        proposal=None,
        certificate=None,
    )


__all__ = [
    "CertificateCallback",
    "CorruptionCertificate",
    "CorruptionConfig",
    "CorruptionProposal",
    "CorruptionSelection",
    "CrystalGeometry",
    "canonical_lattice_from_metric",
    "canonical_request_key",
    "generate_corruption_proposal",
    "lattice_matrix_from_parameters",
    "lattice_parameters_from_matrix",
    "minimum_image_cartesian_retraction",
    "request_seed",
    "select_first_certified_corruption",
    "spd_matrix_power",
    "symmetric_matrix_exp",
    "symmetric_matrix_log",
    "torus_delta",
]

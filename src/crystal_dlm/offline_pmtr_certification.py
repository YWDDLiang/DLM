"""Offline physical certification for deterministic PMTR corruptions.

This module is deliberately potential-agnostic.  Callers inject a batched
EFSM evaluator; no CHGNet, pymatgen, relaxation, or inference-time dependency
is imported here.  Certification uses energies and Cartesian forces at the
same decoded structures plus a small joint SPD/PBC probe.  Stress is retained
only as an EFSM diagnostic because stress and a log-metric tangent expressed
in different lattice gauges must not be contracted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from crystal_dlm.manifold_corruption import (
    CorruptionCertificate,
    CorruptionConfig,
    CorruptionProposal,
    CrystalGeometry,
    canonical_lattice_from_metric,
    spd_congruence_update,
)


Array = np.ndarray


class EFSMBatchEvaluator(Protocol):
    """Potential-independent protocol implemented by offline evaluators."""

    def evaluate(
        self,
        geometries: Sequence[CrystalGeometry],
        *,
        batch_size: int,
    ) -> Sequence[Mapping[str, Any] | None]: ...


@dataclass(frozen=True)
class CertificationConfig:
    """Numerical settings for the local clean-retraction certificate."""

    probe_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < float(self.probe_fraction) < 1.0:
            raise ValueError("probe_fraction must lie strictly between zero and one")


@dataclass(frozen=True)
class EFSMObservation:
    """Validated per-atom energy, Cartesian force, stress, and magnetism."""

    energy_eV_per_atom: float
    forces_eV_per_A: Array
    stress: Array
    magnetic_moments: Array | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        num_sites: int,
    ) -> "EFSMObservation":
        if value is None:
            raise ValueError("EFSM prediction is missing")
        try:
            energy = float(np.asarray(value["e"], dtype=float).reshape(()))
            forces = np.asarray(value["f"], dtype=float)
            stress = _stress_matrix(np.asarray(value["s"], dtype=float))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("EFSM prediction lacks finite e/f/s fields") from exc
        if not math.isfinite(energy):
            raise ValueError("EFSM energy is non-finite")
        if forces.shape != (int(num_sites), 3) or not np.isfinite(forces).all():
            raise ValueError("EFSM forces must be finite with shape [N, 3]")
        if not np.isfinite(stress).all():
            raise ValueError("EFSM stress is non-finite")
        moments = value.get("m")
        magnetic_moments = None
        if moments is not None:
            candidate = np.asarray(moments, dtype=float)
            if candidate.shape not in ((int(num_sites),), (int(num_sites), 1)):
                raise ValueError("EFSM magnetic moments must have one value per site")
            if not np.isfinite(candidate).all():
                raise ValueError("EFSM magnetic moments are non-finite")
            magnetic_moments = candidate.reshape(int(num_sites)).copy()
        return cls(
            energy_eV_per_atom=energy,
            forces_eV_per_A=forces.copy(),
            stress=stress.copy(),
            magnetic_moments=magnetic_moments,
        )

    def summary(self) -> dict[str, float]:
        norms = np.linalg.norm(self.forces_eV_per_A, axis=1)
        return {
            "energy_eV_per_atom": float(self.energy_eV_per_atom),
            "force_rms_eV_per_A": float(np.sqrt(np.mean(norms * norms))),
            "force_max_eV_per_A": float(np.max(norms)),
            "stress_frobenius": float(np.linalg.norm(self.stress)),
        }


def _stress_matrix(stress: Array) -> Array:
    if stress.shape == (3, 3):
        return 0.5 * (stress + stress.T)
    if stress.shape != (6,):
        raise ValueError("stress must be a 3x3 matrix or a six-component vector")
    xx, yy, zz, yz, xz, xy = (float(item) for item in stress)
    return np.asarray(
        ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)), dtype=float
    )


def joint_toward_clean_probe(
    proposal: CorruptionProposal,
    *,
    fraction: float,
) -> CrystalGeometry:
    """Move jointly toward clean along SPD and exact PBC retractions.

    The metric follows the affine congruence geodesic.  Site displacements use
    the exact searched Cartesian minimum-image vectors already frozen in the
    proposal.  They are converted in the corrupted cell and interpolated on
    the torus.  At ``fraction=1`` the sites are PBC-equivalent to clean while
    the metric simultaneously reaches the clean metric.
    """

    alpha = float(fraction)
    if not 0.0 < alpha < 1.0:
        raise ValueError("probe fraction must lie strictly between zero and one")
    corrupted = proposal.geometry
    probe_metric = spd_congruence_update(
        corrupted.metric,
        alpha * np.asarray(proposal.clean_spd_retraction_tangent, dtype=float),
    )
    probe_lattice = canonical_lattice_from_metric(probe_metric)
    fractional_retraction = np.asarray(
        proposal.clean_coordinate_retraction_cartesian, dtype=float
    ) @ np.linalg.inv(corrupted.lattice)
    probe_fractional = np.mod(
        corrupted.frac_coords + alpha * fractional_retraction, 1.0
    )
    return CrystalGeometry(
        lattice=probe_lattice,
        frac_coords=probe_fractional,
        species=corrupted.species,
    )


def _unknown_record(proposal: CorruptionProposal, reason: str) -> dict[str, Any]:
    certificate = CorruptionCertificate(
        post_quantization_valid=False,
        delta_energy=0.0,
        coordinate_force_dot_clean_retraction=None,
        lattice_descent_dot_spd_retraction=None,
    )
    return {
        "schema": "pmtr_corruption_certificate_v1",
        "request_key": proposal.request_key,
        "proposal_index": int(proposal.proposal_index),
        "certified": False,
        "certificate": certificate.to_dict(),
        "evidence": None,
        "failure": str(reason),
    }


def _certificate_record(
    proposal: CorruptionProposal,
    *,
    clean: EFSMObservation,
    corrupted: EFSMObservation,
    probe: EFSMObservation,
    corruption_config: CorruptionConfig,
    certification_config: CertificationConfig,
) -> dict[str, Any]:
    delta_energy = float(
        corrupted.energy_eV_per_atom - clean.energy_eV_per_atom
    )
    coordinate_dot: float | None = None
    if proposal.coordinate_active:
        retraction = np.asarray(
            proposal.clean_coordinate_retraction_cartesian, dtype=float
        )
        coordinate_dot = float(
            np.sum(corrupted.forces_eV_per_A * retraction)
            / len(proposal.geometry.species)
        )
    joint_probe_drop = float(
        corrupted.energy_eV_per_atom - probe.energy_eV_per_atom
    )

    uphill = bool(
        delta_energy > 0.0
        and delta_energy <= float(corruption_config.max_delta_energy)
    )
    coordinate_aligned = bool(
        not proposal.coordinate_active
        or (
            coordinate_dot is not None
            and coordinate_dot > 0.0
        )
    )
    joint_downhill = bool(joint_probe_drop > 0.0)
    post_quantization_valid = bool(
        not proposal.encoding_clipped
        and proposal.has_required_changes(corruption_config)
        and tuple(proposal.geometry.species)
        == tuple(proposal.clean_geometry.species)
    )
    certificate = CorruptionCertificate(
        post_quantization_valid=post_quantization_valid,
        delta_energy=delta_energy,
        coordinate_force_dot_clean_retraction=coordinate_dot,
        # Compatibility field consumed by the existing first-certified
        # builder.  Its value is an observed joint-probe energy drop, never a
        # stress/tangent contraction across lattice gauges.
        lattice_descent_dot_spd_retraction=joint_probe_drop,
    )
    certified = bool(
        post_quantization_valid
        and uphill
        and coordinate_aligned
        and joint_downhill
        and certificate.accepts(proposal, corruption_config)
    )
    return {
        "schema": "pmtr_corruption_certificate_v1",
        "request_key": proposal.request_key,
        "proposal_index": int(proposal.proposal_index),
        "certified": certified,
        "certificate": certificate.to_dict(),
        "evidence": {
            "method": "joint_spd_geodesic_pbc_energy_probe_no_stress_tangent_dot",
            "probe_fraction": float(certification_config.probe_fraction),
            "energy_uphill": uphill,
            "coordinate_force_aligned": coordinate_aligned,
            "joint_probe_downhill": joint_downhill,
            "clean": clean.summary(),
            "corruption": corrupted.summary(),
            "toward_clean_probe": probe.summary(),
        },
        "failure": None if certified else "physical_certificate_rejected",
    }


def certify_corruption_proposals(
    proposals: Sequence[CorruptionProposal],
    *,
    evaluator: EFSMBatchEvaluator,
    corruption_config: CorruptionConfig = CorruptionConfig(),
    certification_config: CertificationConfig = CertificationConfig(),
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Evaluate and certify every proposal without ranking or selection.

    Clean structures are deduplicated by request/body within this call.  Each
    otherwise valid proposal contributes its corrupted structure and one joint
    toward-clean probe.  Result order exactly matches proposal order so a
    downstream builder can apply deterministic first-certified selection.
    """

    if not 8 <= int(batch_size) <= 16:
        raise ValueError("batch_size must be in 8..16")
    proposals = list(proposals)
    if not proposals:
        return []

    geometries: list[CrystalGeometry] = []
    clean_indices: dict[tuple[str, str], int] = {}
    evaluation_indices: list[tuple[int, int, int] | None] = []
    preparation_failures: list[str | None] = []
    for proposal in proposals:
        if not isinstance(proposal, CorruptionProposal):
            raise TypeError("proposals must contain CorruptionProposal values")
        if not proposal.has_required_changes(corruption_config):
            evaluation_indices.append(None)
            preparation_failures.append("proposal_lacks_required_quantized_changes")
            continue
        clean_key = (proposal.request_key, proposal.clean_body)
        clean_index = clean_indices.get(clean_key)
        if clean_index is None:
            clean_index = len(geometries)
            clean_indices[clean_key] = clean_index
            geometries.append(proposal.clean_geometry)
        try:
            probe = joint_toward_clean_probe(
                proposal, fraction=float(certification_config.probe_fraction)
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-proposal failure.
            evaluation_indices.append(None)
            preparation_failures.append(
                f"joint_probe:{type(exc).__name__}:{exc}"[:300]
            )
            continue
        corruption_index = len(geometries)
        geometries.append(proposal.geometry)
        probe_index = len(geometries)
        geometries.append(probe)
        evaluation_indices.append((clean_index, corruption_index, probe_index))
        preparation_failures.append(None)

    raw_observations = list(
        evaluator.evaluate(geometries, batch_size=int(batch_size))
    )
    if len(raw_observations) != len(geometries):
        raise RuntimeError("EFSM evaluator changed batch cardinality")
    observations: list[EFSMObservation | None] = []
    observation_errors: list[str | None] = []
    for geometry, raw in zip(geometries, raw_observations, strict=True):
        try:
            observations.append(
                EFSMObservation.from_mapping(raw, num_sites=len(geometry.species))
            )
            observation_errors.append(None)
        except Exception as exc:  # noqa: BLE001 - retain isolated EFSM failures.
            observations.append(None)
            observation_errors.append(f"{type(exc).__name__}:{exc}"[:300])

    records: list[dict[str, Any]] = []
    for proposal, indices, preparation_failure in zip(
        proposals, evaluation_indices, preparation_failures, strict=True
    ):
        if indices is None:
            records.append(
                _unknown_record(proposal, preparation_failure or "proposal_not_evaluated")
            )
            continue
        clean_index, corruption_index, probe_index = indices
        selected = (
            observations[clean_index],
            observations[corruption_index],
            observations[probe_index],
        )
        if any(item is None for item in selected):
            errors = [
                observation_errors[index]
                for index in indices
                if observation_errors[index] is not None
            ]
            records.append(
                _unknown_record(
                    proposal,
                    "efsm:" + ";".join(errors or ["unknown_prediction"]),
                )
            )
            continue
        clean, corrupted, probe = selected
        assert clean is not None and corrupted is not None and probe is not None
        records.append(
            _certificate_record(
                proposal,
                clean=clean,
                corrupted=corrupted,
                probe=probe,
                corruption_config=corruption_config,
                certification_config=certification_config,
            )
        )
    return records


__all__ = [
    "CertificationConfig",
    "EFSMBatchEvaluator",
    "EFSMObservation",
    "certify_corruption_proposals",
    "joint_toward_clean_probe",
]

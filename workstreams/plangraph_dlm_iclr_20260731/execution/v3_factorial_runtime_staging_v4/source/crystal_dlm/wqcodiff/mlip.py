"""Frozen ASE calculator adapters and relaxation contract for three MLIPs."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping


EVALUATOR_VERSIONS = {
    "chgnet": "0.4.2",
    "mattersim": "1.1.2",
    "mace": "0.3.13",
}
EVALUATOR_CHECKPOINTS = {
    "chgnet": "chgnet-0.3.0.pth.tar",
    "mattersim": "MatterSim-v1.0.0-5M.pth",
    "mace": "mace-mp-0b3-medium.model",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class RelaxationContract:
    optimizer: str = "FIRE"
    relax_positions: bool = True
    relax_cell: bool = True
    fmax_ev_per_angstrom: float = 0.05
    max_steps: int = 500
    maxstep_angstrom: float = 0.1

    def __post_init__(self) -> None:
        if dataclasses.asdict(self) != {
            "optimizer": "FIRE",
            "relax_positions": True,
            "relax_cell": True,
            "fmax_ev_per_angstrom": 0.05,
            "max_steps": 500,
            "maxstep_angstrom": 0.1,
        }:
            raise ValueError("relaxation contract differs from protocol-v3")

    @property
    def sha256(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluatorAsset:
    evaluator: str
    package: str
    package_version: str
    checkpoint: str
    checkpoint_sha256: str
    supported_atomic_numbers: tuple[int, ...]
    support_basis: str
    source_url: str
    license: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluatorAsset":
        return cls(
            evaluator=str(payload["evaluator"]),
            package=str(payload["package"]),
            package_version=str(payload["package_version"]),
            checkpoint=str(payload["checkpoint"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            supported_atomic_numbers=tuple(
                int(value) for value in payload["supported_atomic_numbers"]
            ),
            support_basis=str(payload["support_basis"]),
            source_url=str(payload["source_url"]),
            license=str(payload["license"]),
        )

    def __post_init__(self) -> None:
        if not self.support_basis:
            raise ValueError("evaluator support basis must be recorded")
        if tuple(sorted(set(self.supported_atomic_numbers))) != tuple(
            self.supported_atomic_numbers
        ):
            raise ValueError("supported atomic numbers must be unique and sorted")
        if not self.supported_atomic_numbers or not all(
            1 <= value <= 118 for value in self.supported_atomic_numbers
        ):
            raise ValueError("invalid evaluator element-support set")


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluatorLock:
    path: Path
    assets: Mapping[str, EvaluatorAsset]
    wheelhouse_lock: str
    wheelhouse_lock_sha256: str
    chgnet_dependency_waiver: str
    chgnet_dependency_waiver_sha256: str
    mattersim_dependency_waiver: str
    mattersim_dependency_waiver_sha256: str
    mattersim_runtime_lock: str
    mattersim_runtime_lock_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "EvaluatorLock":
        location = Path(path).resolve()
        payload = json.loads(location.read_text(encoding="utf-8"))
        if payload.get("schema") != "wqcodiff_mlip_asset_lock_v4":
            raise ValueError("invalid MLIP asset-lock schema")
        assets = {
            str(value["evaluator"]): EvaluatorAsset.from_dict(value)
            for value in payload["assets"]
        }
        if set(assets) != set(EVALUATOR_VERSIONS):
            raise ValueError("asset lock must contain exactly CHGNet/MatterSim/MACE")
        wheelhouse_lock = str(payload.get("wheelhouse_lock") or "")
        wheelhouse_lock_sha256 = str(payload.get("wheelhouse_lock_sha256") or "")
        if not wheelhouse_lock or len(wheelhouse_lock_sha256) != 64:
            raise ValueError("asset lock does not pin the offline wheelhouse lock")
        record_fields = (
            "chgnet_dependency_waiver",
            "chgnet_dependency_waiver_sha256",
            "mattersim_dependency_waiver",
            "mattersim_dependency_waiver_sha256",
            "mattersim_runtime_lock",
            "mattersim_runtime_lock_sha256",
        )
        records = {name: str(payload.get(name) or "") for name in record_fields}
        for name, value in records.items():
            if not value or (name.endswith("sha256") and len(value) != 64):
                raise ValueError(f"asset lock does not pin {name}")
        return cls(
            location,
            assets,
            wheelhouse_lock,
            wheelhouse_lock_sha256,
            *(records[name] for name in record_fields),
        )

    def verify(
        self,
        evaluator: str,
        model_root: str | Path,
        *,
        verify_installed: bool = True,
    ) -> EvaluatorAsset:
        root = Path(model_root)
        wheel_lock_path = root / self.wheelhouse_lock
        if not wheel_lock_path.is_file():
            raise FileNotFoundError(f"offline wheelhouse lock is missing: {wheel_lock_path}")
        if sha256_file(wheel_lock_path) != self.wheelhouse_lock_sha256:
            raise RuntimeError("offline wheelhouse lock SHA256 mismatch")
        wheel_payload = json.loads(wheel_lock_path.read_text(encoding="utf-8"))
        if wheel_payload.get("schema") != "wqcodiff_wheelhouse_lock_v4":
            raise ValueError("invalid offline wheelhouse lock schema")
        if wheel_payload.get("stack_id") != "wqcodiff-evaluator-stack-v4":
            raise ValueError("invalid active evaluator-stack ID")
        for filename, expected in (
            (self.chgnet_dependency_waiver, self.chgnet_dependency_waiver_sha256),
            (
                self.mattersim_dependency_waiver,
                self.mattersim_dependency_waiver_sha256,
            ),
            (self.mattersim_runtime_lock, self.mattersim_runtime_lock_sha256),
        ):
            record = root / filename
            if not record.is_file():
                raise FileNotFoundError(f"MLIP dependency record is missing: {record}")
            if sha256_file(record) != expected:
                raise RuntimeError(f"MLIP dependency-record SHA256 mismatch: {record}")
        try:
            asset = self.assets[evaluator]
        except KeyError as exc:
            raise ValueError(f"unknown evaluator {evaluator}") from exc
        expected_version = EVALUATOR_VERSIONS[evaluator]
        if asset.package_version != expected_version:
            raise ValueError(
                f"asset-lock version mismatch for {evaluator}: "
                f"{asset.package_version} != {expected_version}"
            )
        if asset.checkpoint != EVALUATOR_CHECKPOINTS[evaluator]:
            raise ValueError(
                f"asset-lock checkpoint name changed for {evaluator}: {asset.checkpoint}"
            )
        from .vocabulary import MP20_ATOMIC_NUMBERS

        missing_support = sorted(
            set(MP20_ATOMIC_NUMBERS) - set(asset.supported_atomic_numbers)
        )
        if missing_support:
            raise ValueError(
                f"{evaluator} asset lock does not cover MP20 elements: {missing_support}"
            )
        if verify_installed:
            installed = importlib.metadata.version(asset.package)
            if installed != expected_version:
                raise RuntimeError(
                    f"installed {asset.package}=={installed}, expected {expected_version}"
                )
            import torch

            if evaluator == "mattersim":
                from .dependency_waiver import load_mattersim_runtime_lock

                load_mattersim_runtime_lock(
                    root / self.mattersim_runtime_lock,
                    model_root=root,
                    installed_torch=torch.__version__,
                )
            else:
                from .dependency_waiver import (
                    load_chgnet_torch_waiver,
                    validate_mace_runtime_metadata,
                )

                load_chgnet_torch_waiver(
                    root / self.chgnet_dependency_waiver,
                    installed_torch=torch.__version__,
                )
                if evaluator == "mace":
                    validate_mace_runtime_metadata()
        checkpoint = root / asset.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"required {evaluator} checkpoint is missing: {checkpoint}"
            )
        digest = sha256_file(checkpoint)
        if digest != asset.checkpoint_sha256:
            raise RuntimeError(
                f"{evaluator} checkpoint SHA256 mismatch: {digest} != {asset.checkpoint_sha256}"
            )
        return asset


class MLIPCalculator:
    def __init__(
        self,
        *,
        evaluator: str,
        asset_lock: EvaluatorLock,
        model_root: str | Path,
        device: str,
    ) -> None:
        self.evaluator = evaluator
        self.model_root = Path(model_root).resolve()
        self.asset = asset_lock.verify(evaluator, self.model_root)
        self.checkpoint = self.model_root / self.asset.checkpoint
        self.device = device
        self._calculator: Any | None = None
        self.relaxation_contract = RelaxationContract()

    @property
    def contract_hash(self) -> str:
        payload = {
            "evaluator": self.evaluator,
            "package": self.asset.package,
            "package_version": self.asset.package_version,
            "checkpoint_sha256": self.asset.checkpoint_sha256,
            "supported_atomic_numbers": list(self.asset.supported_atomic_numbers),
            "relaxation_sha256": self.relaxation_contract.sha256,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def supported_atomic_numbers(self) -> frozenset[int]:
        return frozenset(self.asset.supported_atomic_numbers)

    def _load(self) -> Any:
        if self._calculator is not None:
            return self._calculator
        if self.evaluator == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator

            self._calculator = CHGNetCalculator.from_file(
                str(self.checkpoint),
                use_device=self.device,
            )
        elif self.evaluator == "mattersim":
            from mattersim.forcefield import MatterSimCalculator

            self._calculator = MatterSimCalculator(
                load_path=str(self.checkpoint),
                device=self.device,
            )
        elif self.evaluator == "mace":
            from mace.calculators import mace_mp

            self._calculator = mace_mp(
                model=str(self.checkpoint),
                device=self.device,
                default_dtype="float32",
                dispersion=False,
            )
        else:  # pragma: no cover - guarded by lock
            raise ValueError(self.evaluator)
        return self._calculator

    def _atoms(self, structure: Any) -> Any:
        from pymatgen.io.ase import AseAtomsAdaptor

        atomic_numbers = {int(value) for value in structure.atomic_numbers}
        unsupported = sorted(atomic_numbers - self.supported_atomic_numbers)
        if unsupported:
            raise ValueError(f"unsupported_elements:{unsupported}")
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self._load()
        return atoms

    def single_point(self, structure: Any) -> dict[str, Any]:
        import numpy as np

        atoms = self._atoms(structure)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(
            np.isfinite(stress)
        ):
            raise RuntimeError("MLIP returned non-finite energy/forces/stress")
        return {
            "energy_total_ev": energy,
            "energy_per_atom_ev": energy / len(atoms),
            "max_force_ev_per_angstrom": float(
                np.max(np.linalg.norm(forces, axis=1))
            ),
            "stress_frobenius_ev_per_a3": float(np.linalg.norm(stress)),
        }

    def relax(self, structure: Any) -> dict[str, Any]:
        import numpy as np
        from ase.optimize import FIRE
        from pymatgen.io.ase import AseAtomsAdaptor

        try:
            from ase.filters import FrechetCellFilter
        except ImportError as exc:
            raise RuntimeError(
                "ASE FrechetCellFilter is required by the frozen cell+position contract"
            ) from exc

        atoms = self._atoms(structure)
        filtered = FrechetCellFilter(atoms)
        optimizer = FIRE(
            filtered,
            logfile=None,
            maxstep=self.relaxation_contract.maxstep_angstrom,
        )
        converged = bool(
            optimizer.run(
                fmax=self.relaxation_contract.fmax_ev_per_angstrom,
                steps=self.relaxation_contract.max_steps,
            )
        )
        steps = int(optimizer.get_number_of_steps())
        if not converged:
            raise RuntimeError(
                f"nonconverged:{steps}/{self.relaxation_contract.max_steps}"
            )
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(
            np.isfinite(stress)
        ):
            raise RuntimeError("MLIP relaxation ended with non-finite outputs")
        relaxed = AseAtomsAdaptor.get_structure(atoms)
        return {
            "energy_total_ev": energy,
            "energy_per_atom_ev": energy / len(atoms),
            "max_force_ev_per_angstrom": float(
                np.max(np.linalg.norm(forces, axis=1))
            ),
            "stress_frobenius_ev_per_a3": float(np.linalg.norm(stress)),
            "relaxation_steps": steps,
            "converged": converged,
            "structure": relaxed.as_dict(),
        }

#!/usr/bin/env python3
"""Run the frozen CrysLLMGen disabled-extension Gate A on one Slurm GPU."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping


def _require_runtime_environment() -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate A CUDA work must run through Slurm")


_require_runtime_environment()

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from crystal_dlm.wqcodiff.crysllmgen.atom_text import (
    parse_upstream_atom_text_fields,
    parse_upstream_atom_text_to_cif,
)
from crystal_dlm.wqcodiff.crysllmgen.disabled_extension import (
    DisabledExtensionRefiner,
)
from crystal_dlm.wqcodiff.crysllmgen.parity import (
    ParityContract,
    audit_parity_report,
    compare_values,
    hash_fixed_select,
    sha256_file,
    write_json_exclusive,
)
from crystal_dlm.wqcodiff.crysllmgen.schedules import (
    build_beta_tables,
    build_coordinate_sigmas,
)


def _relative_hash_manifest(root: Path) -> str:
    records: list[str] = []
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix != ".pyc"
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        records.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest()


def _load_function_from_source(path: Path, function_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise RuntimeError(f"{function_name} is absent from {path}")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Structure": Structure, "Lattice": Lattice, "np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _load_diff_utils(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("wq_gate_a_upstream_diff_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream diff_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_selected_rows(path: Path, contract: ParityContract) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return hash_fixed_select(
        rows,
        identity=lambda row: row["material_id"],
        count=contract.proposal_count,
        salt=contract.selection_salt,
    )


def _structure_to_atom_text(cif: str) -> str:
    structure = Structure.from_str(cif, fmt="cif")
    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    lines = [
        " ".join(f"{float(value):.1f}" for value in lengths),
        " ".join(str(int(value)) for value in angles),
    ]
    for species, coordinate in zip(structure.species, structure.frac_coords):
        lines.append(str(species))
        lines.append(" ".join(f"{float(value):.2f}" for value in coordinate))
    return "\n".join(lines)


def _corrupt_parser_case(text: str, ordinal: int) -> str:
    if ordinal % 8:
        return text
    mode = (ordinal // 8) % 4
    lines = text.splitlines()
    if mode == 0:
        return "answer follows\n" + text
    if mode == 1:
        return "\n".join(lines[:-1])
    if mode == 2:
        lines[0] = "not-a-number " + " ".join(lines[0].split()[1:])
        return "\n".join(lines)
    return text + "\ntrailing prose"


def _cif_semantics(cif: str) -> dict[str, Any]:
    structure = Structure.from_str(cif, fmt="cif")
    return {
        "lengths": list(structure.lattice.abc),
        "angles": list(structure.lattice.angles),
        "species": [str(value) for value in structure.species],
        "frac_coords": (structure.frac_coords % 1.0).tolist(),
    }


def _parser_check(
    rows: list[dict[str, str]], upstream_parser: Any, contract: ParityContract
) -> tuple[dict[str, Any], list[str]]:
    comparisons = []
    failed_inputs = 0
    selected_ids: list[str] = []
    for ordinal, row in enumerate(rows):
        selected_ids.append(row["material_id"])
        raw = _corrupt_parser_case(_structure_to_atom_text(row["cif"]), ordinal)
        upstream_error = derived_error = None
        upstream_cif = derived_cif = None
        try:
            upstream_cif = upstream_parser(raw)
        except Exception as exc:  # failures are observations in the denominator.
            upstream_error = type(exc).__name__
        try:
            derived_cif = parse_upstream_atom_text_to_cif(raw)
        except Exception as exc:
            derived_error = type(exc).__name__
        if upstream_error is not None or derived_error is not None:
            failed_inputs += 1
            comparison = compare_values(
                {"failed": upstream_error is not None},
                {"failed": derived_error is not None},
                absolute_tolerance=contract.absolute_tolerance,
                relative_tolerance=contract.relative_tolerance,
            )
        else:
            comparison = compare_values(
                _cif_semantics(str(upstream_cif)),
                _cif_semantics(str(derived_cif)),
                absolute_tolerance=contract.absolute_tolerance,
                relative_tolerance=contract.relative_tolerance,
            )
        comparisons.append(comparison)
    maximum = max(value.max_absolute_error for value in comparisons)
    first = next((value.first_mismatch for value in comparisons if not value.passed), None)
    return (
        {
            "passed": all(value.passed for value in comparisons),
            "max_absolute_error": maximum,
            "cases": len(comparisons),
            "synthetic_failure_cases": failed_inputs,
            "first_mismatch": first,
        },
        selected_ids,
    )


def _load_upstream_diffusion(snapshot_root: Path) -> Any:
    sys.path.insert(0, str(snapshot_root))
    try:
        from models_ddpm.diffusion import CSPDiffusion
    finally:
        # Imported modules keep their resolved locations; avoid leaking the path
        # into later subprocess/module resolution.
        sys.path.pop(0)
    return CSPDiffusion


class _Batch(types.SimpleNamespace):
    def to(self, device: torch.device) -> "_Batch":
        return _Batch(
            num_graphs=self.num_graphs,
            num_atoms=self.num_atoms.to(device),
            atom_types=self.atom_types.to(device),
            frac_coords=self.frac_coords.to(device),
            lengths=self.lengths.to(device),
            angles=self.angles.to(device),
            batch=self.batch.to(device),
        )


def _make_batch(rows: list[dict[str, str]], count: int) -> _Batch:
    structures = [Structure.from_str(row["cif"], fmt="cif") for row in rows[:count]]
    num_atoms = torch.tensor([len(value) for value in structures], dtype=torch.long)
    atom_types = torch.tensor(
        [number for value in structures for number in value.atomic_numbers],
        dtype=torch.long,
    )
    coordinates = torch.tensor(
        np.concatenate([value.frac_coords for value in structures], axis=0),
        dtype=torch.float32,
    )
    lengths = torch.tensor([value.lattice.abc for value in structures], dtype=torch.float32)
    angles = torch.tensor(
        [value.lattice.angles for value in structures], dtype=torch.float32
    )
    node_to_graph = torch.repeat_interleave(torch.arange(count), num_atoms)
    return _Batch(
        num_graphs=count,
        num_atoms=num_atoms,
        atom_types=atom_types,
        frac_coords=coordinates,
        lengths=lengths,
        angles=angles,
        batch=node_to_graph,
    )


def _tensor_payload(values: Any) -> Any:
    if isinstance(values, Mapping):
        return {key: _tensor_payload(value) for key, value in values.items()}
    if isinstance(values, (tuple, list)):
        return [_tensor_payload(value) for value in values]
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return values


def _seed(seed: int) -> None:
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_checks(
    rows: list[dict[str, str]],
    snapshot_root: Path,
    checkpoint_path: Path,
    contract: ParityContract,
    reverse_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    CSPDiffusion = _load_upstream_diffusion(snapshot_root)
    sampler_device = torch.device("cuda")
    comparison_device = torch.device(contract.one_step_device)
    model = CSPDiffusion(contract.scheduler_timesteps, contract.parent_run_type)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=True)
    model.eval()
    mapping_check = {
        "passed": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "max_absolute_error": 0.0,
        "checkpoint_keys": len(state),
        "model_keys": len(model.state_dict()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameters": sum(value.numel() for value in model.parameters()),
    }
    wrapper = DisabledExtensionRefiner(model)
    batch8 = _make_batch(rows, 8).to(comparison_device)
    times = torch.tensor(
        [1, 17, 64, 128, 256, 400, 600, 800], device=comparison_device
    )
    embeddings = model.time_embedding(times)
    lattices = __import__(
        "models_ddpm.data_utils", fromlist=["lattice_params_to_matrix_torch"]
    ).lattice_params_to_matrix_torch(batch8.lengths, batch8.angles)
    with torch.no_grad():
        direct = model.decoder(
            embeddings,
            batch8.atom_types,
            batch8.frac_coords,
            lattices,
            batch8.num_atoms,
            batch8.batch,
        )
        wrapped = wrapper.decoder_step(
            embeddings,
            batch8.atom_types,
            batch8.frac_coords,
            lattices,
            batch8.num_atoms,
            batch8.batch,
        )
    tensor_comparison = compare_values(
        _tensor_payload(direct),
        _tensor_payload(wrapped),
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    tensor_check = {
        **tensor_comparison.to_dict(),
        "batch_size": 8,
        "device": str(comparison_device),
        "cuda_atomic_scatter_avoided": True,
    }

    model = model.to(sampler_device)
    batch4 = _make_batch(rows, 4).to(sampler_device)
    _seed(2026072001)
    with torch.no_grad():
        direct_sample, _ = model.sample(batch4, step_lr=1.0e-5, diff_steps=reverse_steps)
    _seed(2026072001)
    with torch.no_grad():
        wrapped_sample, _ = wrapper.sample(
            batch4, step_lr=1.0e-5, diff_steps=reverse_steps
        )
    sampler_comparison = compare_values(
        _tensor_payload(direct_sample),
        _tensor_payload(wrapped_sample),
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    sampler_check = {
        **sampler_comparison.to_dict(),
        "batch_size": 4,
        "reverse_steps": reverse_steps,
        "csp_forwards_per_sample": 2 * reverse_steps,
    }
    return mapping_check, tensor_check, sampler_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reverse-steps", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("Gate A requires a CUDA allocation")
    project_root = args.project_root.resolve()
    contract = ParityContract.load(args.contract)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_parent = project_root / "crystal_dlm/wqcodiff/crysllmgen"
    snapshot_root = snapshot_parent / "upstream"
    snapshot_manifest = json.loads(
        (snapshot_parent / "UPSTREAM_MANIFEST.json").read_text(encoding="utf-8")
    )
    observed_snapshot_sha = _relative_hash_manifest(snapshot_root)
    source_check = {
        "passed": observed_snapshot_sha
        == contract.upstream_relative_manifest_sha256
        == snapshot_manifest["relative_file_hash_manifest_sha256"]
        and snapshot_manifest["upstream_commit"] == contract.upstream_commit,
        "max_absolute_error": 0.0,
        "observed_relative_manifest_sha256": observed_snapshot_sha,
        "upstream_commit": snapshot_manifest["upstream_commit"],
        "source_file_count": snapshot_manifest["source_file_count"],
    }
    rows = _load_selected_rows(args.validation_csv.resolve(), contract)
    upstream_parser = _load_function_from_source(
        snapshot_root / "crysllmgen_sample.py", "parse_generated_text"
    )
    parser_check, selected_ids = _parser_check(rows, upstream_parser, contract)
    diff_utils = _load_diff_utils(snapshot_root / "models_ddpm/diff_utils.py")
    upstream_beta = diff_utils.BetaScheduler(contract.scheduler_timesteps, "cosine")
    _seed(2026072000)
    upstream_coordinate = diff_utils.SigmaScheduler(
        contract.scheduler_timesteps, 0.005, 0.5
    )
    derived_beta = build_beta_tables(contract.scheduler_timesteps)
    beta_comparison = compare_values(
        {
            "betas": upstream_beta.betas,
            "alphas": upstream_beta.alphas,
            "alphas_cumprod": upstream_beta.alphas_cumprod,
            "posterior_sigmas": upstream_beta.sigmas,
            "coordinate_sigmas": upstream_coordinate.sigmas,
        },
        {
            **derived_beta,
            "coordinate_sigmas": build_coordinate_sigmas(
                contract.scheduler_timesteps
            ),
        },
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    schedule_check = {
        **beta_comparison.to_dict(),
        "scheduler_timesteps": contract.scheduler_timesteps,
    }
    mapping_check, tensor_check, sampler_check = _model_checks(
        rows,
        snapshot_root,
        args.checkpoint.resolve(),
        contract,
        args.reverse_steps,
    )
    attempts_check = {
        "passed": len(selected_ids) == contract.proposal_count,
        "max_absolute_error": 0.0,
        "submitted": len(selected_ids),
        "terminal": len(selected_ids),
        "retry_or_replacement_used": False,
        "unique_ids": len(set(selected_ids)),
    }
    checks: OrderedDict[str, dict[str, Any]] = OrderedDict(
        (
            ("source_snapshot", source_check),
            ("atom_parser", parser_check),
            ("beta_sigma_tables", schedule_check),
            ("checkpoint_mapping", mapping_check),
            ("one_step_csp_tensors", tensor_check),
            ("deterministic_sampler", sampler_check),
            ("attempt_accounting", attempts_check),
        )
    )
    report = {
        "schema": "crysllmgen_disabled_extension_parity_report_v1",
        "contract_sha256": contract.sha256,
        "proposal_count": contract.proposal_count,
        "terminal_attempts": len(selected_ids),
        "retry_or_replacement_used": False,
        "checks": checks,
        "selected_material_ids_sha256": hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "runtime": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": sys.version,
            "walltime_s": time.time() - started,
            "blas_threads": 1,
            "offline": True,
        },
    }
    audit = audit_parity_report(report, contract)
    write_json_exclusive(output / "selected_material_ids.json", {"ids": selected_ids})
    write_json_exclusive(output / "parity_report.json", report)
    write_json_exclusive(output / "parity_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()

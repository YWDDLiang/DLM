#!/usr/bin/env python3
"""Generate one frozen Planner arm with paired R5-C and parent noise."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = HERE.parents[3]
for location in (PROJECT_ROOT_FALLBACK, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from crystal_dlm.fixed_slot import MASK_TOKEN_ID  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.r5_plan_state import build_body_prompt  # noqa: E402
from paired_llada import generate_paired_exact_plan  # noqa: E402
from paired_noise import paired_randn_bank  # noqa: E402
from protocol import (  # noqa: E402
    canonical_sha256,
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_sha,
    require_source_manifest,
    resolve_project_path,
    sha256_file,
    validate_arm,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from scripts.sample_llada_dynamic_crystals import (  # noqa: E402
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import (  # noqa: E402
    element_prefill_for_batch,
    merge_prefill_maps,
)


def _require_runtime() -> torch.device:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("paired body/refinement must run through Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu":
        raise RuntimeError("paired body/refinement requires the gpu partition")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if cpus != 8:
        raise RuntimeError("paired body/refinement requires exactly eight CPUs")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if int(os.environ.get(name, "0")) != cpus:
            raise RuntimeError(f"{name} must equal the allocated CPU count")
    if os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff":
        raise RuntimeError("paired body/refinement requires diff_meets_diff")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("paired body/refinement requires exactly one CUDA device")
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError(f"paired body/refinement requires A800, observed {name}")
    return torch.device("cuda", 0)


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _verify_directory_sizes(manifest: Mapping[str, Any]) -> None:
    root = Path(str(manifest["path"])).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    observed = {
        str(item.relative_to(root)): item.stat().st_size
        for item in root.rglob("*")
        if item.is_file()
    }
    expected = {
        str(item["relative_path"]): int(item["bytes"]) for item in manifest["files"]
    }
    if observed != expected:
        raise ValueError(f"model directory file/size inventory changed: {root}")
    for item in manifest["files"]:
        if int(item["bytes"]) <= 64 * 1024 * 1024:
            require_sha(
                root / str(item["relative_path"]),
                str(item["sha256"]),
                f"model metadata {item['relative_path']}",
            )


def _load_contracts(
    *,
    config_path: Path,
    source_dir: Path,
    project_root: Path,
    data_dir: Path,
    execution_sha: str,
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    require_source_manifest(source_dir, execution_sha)
    require_runtime_manifest(project_root, source_dir)
    config = read_json(config_path)
    if config.get("identity") != "h1a2c_p0_p1_sun256_exploratory_v1":
        raise ValueError("generation protocol identity changed")
    if not (data_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(data_dir / "_SUCCESS")
    manifest = read_json(data_dir / "ledger_manifest.json")
    if (
        manifest.get("ok") is not True
        or manifest.get("execution_manifest_sha256") != execution_sha
        or int(manifest.get("pairs", -1)) != 256
        or int(manifest.get("all_attempt_denominator_per_arm", -1)) != 256
        or manifest.get("retry_or_replacement_used") is not False
    ):
        raise ValueError("paired ledger manifest contract changed")
    require_sha(
        data_dir / "attempt_ledger.jsonl",
        manifest["attempt_ledger"]["sha256"],
        "paired attempt ledger",
    )
    require_sha(
        data_dir / "asset_manifest.json",
        manifest["asset_manifest"]["sha256"],
        "paired asset manifest",
    )
    if sha256_file(config_path) != manifest["config"]["sha256"]:
        raise ValueError("generation config differs from frozen ledger config")
    ledger = read_jsonl(data_dir / "attempt_ledger.jsonl")
    if (
        len(ledger) != 256
        or [int(row.get("ordinal", -1)) for row in ledger] != list(range(256))
        or any(arm not in row.get("arms", {}) for row in ledger)
    ):
        raise ValueError("paired ledger rows changed")
    asset_manifest = read_json(data_dir / "asset_manifest.json")
    if asset_manifest.get("execution_manifest_sha256") != execution_sha:
        raise ValueError("asset manifest source identity changed")
    assets = asset_manifest["assets"]
    _verify_directory_sizes(assets["body_base_model"])
    _verify_directory_sizes(assets["body_checkpoint"])
    require_sha(
        Path(assets["body_adapter"]["path"]),
        assets["body_adapter"]["sha256"],
        "R5-C body adapter",
    )
    require_sha(
        Path(assets["parent_checkpoint"]["path"]),
        config["parent_refiner"]["checkpoint_sha256"],
        "CrysLLMGen parent checkpoint",
    )
    return config, ledger, asset_manifest


def _body_stage(
    *,
    arm: str,
    config: dict[str, Any],
    ledger: list[dict[str, Any]],
    project_root: Path,
    device: torch.device,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm_spec = config["source_plan_run"]["arms"][arm]
    plan_path = resolve_project_path(project_root, arm_spec["raw_generations"])
    require_sha(
        plan_path,
        arm_spec["raw_generations_sha256"],
        f"{arm} frozen Planner output",
    )
    source_rows = read_jsonl(plan_path)
    if len(source_rows) != 512 or [
        int(row.get("sample_idx", -1)) for row in source_rows
    ] != list(range(512)):
        raise ValueError("frozen Planner output order changed")
    tasks: list[dict[str, Any]] = []
    body_records: dict[int, dict[str, Any]] = {}
    for cell in ledger:
        ordinal = int(cell["ordinal"])
        entry = cell["arms"][arm]
        source_row = source_rows[ordinal]
        if canonical_sha256(source_row) != entry["source_record_sha256"]:
            raise ValueError(f"{arm} source Planner row changed at {ordinal}")
        base_record = {
            "schema": "h1a2c_p0_p1_body_attempt_v1",
            "arm": arm,
            "ordinal": ordinal,
            "sample_idx": int(cell["sample_idx"]),
            "pair_id": cell["pair_id"],
            "attempt_id": entry["attempt_id"],
            "method": entry["method"],
            "body_noise_seed": int(cell["body_noise_seed"]),
            "refiner_noise_seed": int(cell["refiner_noise_seed"]),
            "plan_state_sha256": entry["plan_state_sha256"],
            "retry_or_replacement_used": False,
        }
        if not entry["body_eligible"]:
            body_records[ordinal] = {
                **base_record,
                "status": "failed",
                "reason": str(entry["ineligible_reason"]),
                "text": "",
                "arrays": None,
            }
            continue
        plan = dict(entry["plan_state"])
        tasks.append(
            {
                **base_record,
                "plan_state": plan,
                "prompt": build_body_prompt(plan).rstrip() + "\n",
            }
        )

    crysllmgen_dir = resolve_project_path(
        project_root, config["parent_refiner"]["crysllmgen_snapshot"]
    )
    process_one = import_process_one(crysllmgen_dir)
    body = config["body"]
    model, tokenizer = load_model_and_tokenizer(
        str(resolve_project_path(project_root, body["base_model"])),
        str(resolve_project_path(project_root, body["checkpoint"])),
        device,
    )
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer pad token collides with LLaDA mask token")
    tokenizer_report = {
        "schema": "h1a2c_p0_p1_body_tokenizer_v1",
        "arm": arm,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "mask_token_id": MASK_TOKEN_ID,
    }
    write_json_exclusive(output / "tokenizer_report.json", tokenizer_report)

    proposal_records: list[dict[str, Any]] = []
    tasks.sort(key=lambda item: (int(item["plan_state"]["N"]), int(item["ordinal"])))
    batch_size = int(body["batch_size"])
    offset = 0
    started = time.monotonic()
    with tqdm(total=len(tasks), desc=f"{arm} paired R5-C body") as progress:
        while offset < len(tasks):
            num_atoms = int(tasks[offset]["plan_state"]["N"])
            batch: list[dict[str, Any]] = []
            while (
                offset < len(tasks)
                and len(batch) < batch_size
                and int(tasks[offset]["plan_state"]["N"]) == num_atoms
            ):
                batch.append(tasks[offset])
                offset += 1
            prompts = [item["prompt"] for item in batch]
            gen_length = exact_body_token_count(num_atoms)
            allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
            prefill = merge_prefill_maps(
                count_prefill_for_batch(tokenizer, num_atoms, len(batch)),
                element_prefill_for_batch(
                    tokenizer, [item["plan_state"] for item in batch]
                ),
            )
            schedule = exact_dynamic_generation_schedule(num_atoms)
            lightweight = build_dynamic_lightweight_constraints(
                tokenizer,
                duplicate_coordinate_mask=bool(body["duplicate_coordinate_mask"]),
                lattice_volume_mask=bool(body["lattice_volume_mask"]),
                min_lattice_rad=float(body["min_lattice_rad"]),
            )
            encoded = tokenizer(
                prompts, add_special_tokens=False, padding=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(_model_device(model))
            attention_mask = encoded["attention_mask"].to(_model_device(model))
            outputs = generate_paired_exact_plan(
                model,
                input_ids,
                base_seeds=[int(item["body_noise_seed"]) for item in batch],
                attention_mask=attention_mask,
                gen_length=gen_length,
                temperature=float(body["temperature"]),
                cfg_scale=float(body["cfg_scale"]),
                remasking=str(body["remasking"]),
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed,
                prefill_token_ids_by_generation_pos=prefill,
                generation_position_groups=schedule,
                lightweight_decoding_constraints=lightweight,
            )
            generated = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for item, text in zip(batch, decoded):
                ordinal = int(item["ordinal"])
                record = {
                    key: value
                    for key, value in item.items()
                    if key not in {"plan_state", "prompt"}
                }
                record.update({"status": "failed", "reason": "", "text": text})
                try:
                    arrays = validate_answer_matches_plan(item["plan_state"], text)
                    graph, cif = graph_from_arrays(arrays, process_one)
                    record.update(
                        {
                            "status": "succeeded",
                            "reason": "",
                            "arrays": arrays,
                            "proposal_cif_sha256": canonical_sha256(cif),
                        }
                    )
                    proposal_records.append(
                        {
                            "ordinal": ordinal,
                            "attempt_id": item["attempt_id"],
                            "pair_id": item["pair_id"],
                            "refiner_noise_seed": int(item["refiner_noise_seed"]),
                            "graph": graph,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    record.update(
                        {
                            "status": "failed",
                            "reason": f"body:{type(exc).__name__}:{exc}",
                            "arrays": None,
                        }
                    )
                body_records[ordinal] = record
                progress.update(1)

    ordered = [body_records[index] for index in range(256)]
    write_jsonl_exclusive(output / "body_attempts.jsonl", ordered)
    with (output / "proposal_graphs.pt").open("xb") as handle:
        torch.save(proposal_records, handle)
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": "h1a2c_p0_p1_body_report_v1",
        "ok": True,
        "arm": arm,
        "attempts": 256,
        "planner_or_body_ineligible": sum(
            row["status"] != "succeeded" for row in ordered
        ),
        "graph_succeeded": len(proposal_records),
        "body_attempts_sha256": sha256_file(output / "body_attempts.jsonl"),
        "proposal_graphs_sha256": sha256_file(output / "proposal_graphs.pt"),
        "walltime_s": time.monotonic() - started,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(output / "body_report.json", report)
    del model, tokenizer, process_one
    gc.collect()
    torch.cuda.empty_cache()
    return ordered, proposal_records


def _setup_parent_imports(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from models_ddpm.data_utils import lattice_params_to_matrix_torch
    from models_ddpm.diffusion import CSPDiffusion
    from torch_geometric.data import Data, DataLoader

    return CSPDiffusion, Data, DataLoader, lattice_params_to_matrix_torch


class _ProposalDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]], data_class: Any):
        self.records = records
        self.data_class = data_class

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        source = self.records[index]["graph"]
        num_atoms = int(torch.as_tensor(source["n_atom"]).view(-1)[0].item())
        return self.data_class(
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=num_atoms,
            num_bonds=source["edge_indices"].shape[0],
            lengths=torch.as_tensor(source["length"], dtype=torch.float32).view(1, 3),
            angles=torch.as_tensor(source["angle"], dtype=torch.float32).view(1, 3),
            frac_coords=torch.as_tensor(source["x_coord"], dtype=torch.float32),
            atom_types=torch.LongTensor(source["a_type"]),
            edge_index=torch.LongTensor(source["edge_indices"].T).contiguous(),
            to_jimages=torch.LongTensor(source["to_jimages"]),
        )


@torch.no_grad()
def _paired_parent_sample(
    model: Any,
    batch: Any,
    *,
    base_seeds: list[int],
    diff_steps: int,
    max_atoms: int,
    lattice_params_to_matrix_torch: Any,
) -> dict[str, torch.Tensor]:
    if len(base_seeds) != int(batch.num_graphs):
        raise ValueError("one parent-noise seed is required per graph")
    if int(diff_steps) != 800:
        raise ValueError("frozen parent refinement requires exactly 800 steps")
    num_atoms = [int(value) for value in batch.num_atoms.detach().cpu().view(-1)]
    if any(value < 1 or value > max_atoms for value in num_atoms):
        raise ValueError("parent input atom count is outside the max-20 noise domain")
    x_state = batch.frac_coords
    lattice_state = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    coord_corrector = paired_randn_bank(
        base_seeds,
        role="coord_corrector",
        diffusion_steps=diff_steps,
        trailing_shape=(max_atoms, 3),
        device=x_state.device,
        dtype=x_state.dtype,
    )
    coord_predictor = paired_randn_bank(
        base_seeds,
        role="coord_predictor",
        diffusion_steps=diff_steps,
        trailing_shape=(max_atoms, 3),
        device=x_state.device,
        dtype=x_state.dtype,
    )
    lattice_predictor = paired_randn_bank(
        base_seeds,
        role="lattice_predictor",
        diffusion_steps=diff_steps,
        trailing_shape=(3, 3),
        device=lattice_state.device,
        dtype=lattice_state.dtype,
    )

    def atom_noise(bank: torch.Tensor, noise_index: int) -> torch.Tensor:
        return torch.cat(
            [
                bank[row_index, noise_index, :atom_count]
                for row_index, atom_count in enumerate(num_atoms)
            ],
            dim=0,
        )

    for timestep in tqdm(
        range(diff_steps, 0, -1),
        desc="paired CrysLLMGen parent",
        leave=False,
    ):
        times = torch.full((int(batch.num_graphs),), timestep, device=model.device)
        time_embedding = model.time_embedding(times)
        alpha = model.beta_scheduler.alphas[timestep]
        alpha_cumprod = model.beta_scheduler.alphas_cumprod[timestep]
        sigma_lattice = model.beta_scheduler.sigmas[timestep]
        sigma_coord = model.sigma_scheduler.sigmas[timestep]
        sigma_norm = model.sigma_scheduler.sigmas_norm[timestep]
        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        if timestep > 1:
            noise_index = diff_steps - timestep
            correction_noise = atom_noise(coord_corrector, noise_index)
        else:
            noise_index = -1
            correction_noise = torch.zeros_like(x_state)
        correction_step = 1e-5 * (sigma_coord / model.sigma_scheduler.sigma_begin) ** 2
        correction_std = torch.sqrt(2 * correction_step)
        _, predicted_coord = model.decoder(
            time_embedding,
            batch.atom_types,
            x_state,
            lattice_state,
            batch.num_atoms,
            batch.batch,
        )
        predicted_coord = predicted_coord * torch.sqrt(sigma_norm)
        corrected_coords = (
            x_state
            - correction_step * predicted_coord
            + correction_std * correction_noise
        )
        if timestep > 1:
            prediction_coord_noise = atom_noise(coord_predictor, noise_index)
            prediction_lattice_noise = lattice_predictor[:, noise_index]
        else:
            prediction_coord_noise = torch.zeros_like(x_state)
            prediction_lattice_noise = torch.zeros_like(lattice_state)
        adjacent_sigma = model.sigma_scheduler.sigmas[timestep - 1]
        prediction_step = sigma_coord**2 - adjacent_sigma**2
        prediction_std = torch.sqrt(
            adjacent_sigma**2 * (sigma_coord**2 - adjacent_sigma**2) / sigma_coord**2
        )
        predicted_lattice, predicted_coord = model.decoder(
            time_embedding,
            batch.atom_types,
            corrected_coords,
            lattice_state,
            batch.num_atoms,
            batch.batch,
        )
        predicted_coord = predicted_coord * torch.sqrt(sigma_norm)
        x_state = (
            corrected_coords
            - prediction_step * predicted_coord
            + prediction_std * prediction_coord_noise
        ) % 1.0
        lattice_state = (
            c0 * (lattice_state - c1 * predicted_lattice)
            + sigma_lattice * prediction_lattice_noise
        )
    return {
        "num_atoms": batch.num_atoms,
        "atom_types": batch.atom_types,
        "frac_coords": x_state,
        "lattices": lattice_state,
    }


def _structure_from_output(
    output: Mapping[str, torch.Tensor],
    graph_index: int,
    atom_offset: int,
) -> tuple[dict[str, Any], int]:
    from pymatgen.core import Lattice, Structure

    num_atoms = int(output["num_atoms"][graph_index].detach().cpu().item())
    stop = atom_offset + num_atoms
    coords = output["frac_coords"][atom_offset:stop].detach().cpu().numpy()
    atom_types = [
        int(value)
        for value in output["atom_types"][atom_offset:stop].detach().cpu().tolist()
    ]
    matrix = output["lattices"][graph_index].detach().cpu().numpy()
    scalars = [*coords.reshape(-1).tolist(), *matrix.reshape(-1).tolist()]
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("nonfinite parent output")
    structure = Structure(
        Lattice(np.asarray(matrix, dtype=float)),
        atom_types,
        np.asarray(coords, dtype=float),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )
    if structure.num_sites != num_atoms or structure.volume < 0.1:
        raise ValueError("unsupported parent output lattice or site count")
    return structure.as_dict(), stop


def _parent_stage(
    *,
    arm: str,
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
    body_records: list[dict[str, Any]],
    proposal_records: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    parent = config["parent_refiner"]
    crysllmgen_dir = resolve_project_path(project_root, parent["crysllmgen_snapshot"])
    CSPDiffusion, Data, DataLoader, lattice_converter = _setup_parent_imports(
        crysllmgen_dir
    )
    dataset = _ProposalDataset(proposal_records, Data)
    dataloader = DataLoader(
        dataset,
        batch_size=int(parent["batch_size"]),
        shuffle=False,
    )
    model = CSPDiffusion(int(parent["timesteps"]), "train").to(device)
    model.device = device
    checkpoint_path = resolve_project_path(project_root, parent["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    structure_by_ordinal: dict[int, dict[str, Any]] = {}
    parent_failure_by_ordinal: dict[int, str] = {}
    started = time.monotonic()
    record_offset = 0
    for batch in dataloader:
        batch_count = int(batch.num_graphs)
        metadata = proposal_records[record_offset : record_offset + batch_count]
        record_offset += batch_count
        batch = batch.to(device)
        results = _paired_parent_sample(
            model,
            batch,
            base_seeds=[int(item["refiner_noise_seed"]) for item in metadata],
            diff_steps=int(parent["diffusion_steps"]),
            max_atoms=int(parent["maximum_atoms_for_common_noise"]),
            lattice_params_to_matrix_torch=lattice_converter,
        )
        atom_offset = 0
        for graph_index, item in enumerate(metadata):
            ordinal = int(item["ordinal"])
            try:
                structure, atom_offset = _structure_from_output(
                    results, graph_index, atom_offset
                )
                structure_by_ordinal[ordinal] = structure
            except Exception as exc:  # noqa: BLE001
                num_atoms = int(results["num_atoms"][graph_index].detach().cpu().item())
                atom_offset += num_atoms
                parent_failure_by_ordinal[ordinal] = (
                    f"parent:{type(exc).__name__}:{exc}"
                )
    if record_offset != len(proposal_records):
        raise RuntimeError("parent DataLoader omitted proposal records")

    method = str(config["source_plan_run"]["arms"][arm]["method"])
    generation_rows = []
    for ordinal, body_record in enumerate(body_records):
        status = "failed"
        reason = str(body_record.get("reason", "body_failed"))
        structure = None
        if body_record["status"] == "succeeded":
            if ordinal in structure_by_ordinal:
                status = "succeeded"
                reason = ""
                structure = structure_by_ordinal[ordinal]
            else:
                reason = parent_failure_by_ordinal.get(ordinal, "parent:missing_output")
        generation_rows.append(
            {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": body_record["attempt_id"],
                "method": method,
                "ordinal": ordinal,
                "sample_idx": int(body_record["sample_idx"]),
                "pair_id": body_record["pair_id"],
                "status": status,
                "reason": reason,
                "structure": structure,
                "body_noise_seed": int(body_record["body_noise_seed"]),
                "refiner_noise_seed": int(body_record["refiner_noise_seed"]),
                "source_plan_state_sha256": body_record["plan_state_sha256"],
                "retry_or_replacement_used": False,
            }
        )
    generation_path = output / "generation.jsonl"
    write_jsonl_exclusive(generation_path, generation_rows)
    report = {
        "schema": "h1a2c_p0_p1_generation_report_v1",
        "ok": True,
        "arm": arm,
        "method": method,
        "attempts": 256,
        "generation_succeeded": sum(
            row["status"] == "succeeded" for row in generation_rows
        ),
        "generation_failed": sum(row["status"] == "failed" for row in generation_rows),
        "body_graph_succeeded": len(proposal_records),
        "parent_structure_succeeded": len(structure_by_ordinal),
        "generation_jsonl_sha256": sha256_file(generation_path),
        "diffusion_steps": int(parent["diffusion_steps"]),
        "common_noise_max_atoms": int(parent["maximum_atoms_for_common_noise"]),
        "walltime_s": time.monotonic() - started,
        "retry_or_replacement_used": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(output / "generation_report.json", report)
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return generation_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "execution source manifest"
    )
    device = _require_runtime()
    project_root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    config, ledger, asset_manifest = _load_contracts(
        config_path=args.config.resolve(),
        source_dir=args.source_dir.resolve(),
        project_root=project_root,
        data_dir=args.data_dir.resolve(),
        execution_sha=execution_sha,
        arm=arm,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(
        output / "run_contract.json",
        {
            "schema": "h1a2c_p0_p1_generation_contract_v1",
            "arm": arm,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "execution_manifest_sha256": execution_sha,
            "asset_manifest_sha256": sha256_file(
                args.data_dir.resolve() / "asset_manifest.json"
            ),
            "attempts": 256,
            "retry_or_replacement_used": False,
            "automatic_downstream_authorized": False,
        },
    )
    body_records, proposal_records = _body_stage(
        arm=arm,
        config=config,
        ledger=ledger,
        project_root=project_root,
        device=device,
        output=output,
    )
    generation_rows = _parent_stage(
        arm=arm,
        config=config,
        project_root=project_root,
        device=device,
        body_records=body_records,
        proposal_records=proposal_records,
        output=output,
    )
    if (
        len(generation_rows) != 256
        or [int(row["ordinal"]) for row in generation_rows] != list(range(256))
        or len({row["attempt_id"] for row in generation_rows}) != 256
        or any(row["retry_or_replacement_used"] is not False for row in generation_rows)
    ):
        raise RuntimeError("generation terminal denominator validation failed")
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "arm": arm,
                "attempts": 256,
                "succeeded": sum(
                    row["status"] == "succeeded" for row in generation_rows
                ),
                "failed": sum(row["status"] == "failed" for row in generation_rows),
                "execution_manifest_sha256": execution_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

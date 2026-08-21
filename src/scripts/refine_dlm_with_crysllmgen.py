#!/usr/bin/env python3
"""Run CrysLLMGen diffusion refinement on DLM proposal graphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm


def setup_crysllmgen_imports(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from config import config
    from models_ddpm.diffusion import CSPDiffusion
    from torch_geometric.data import Data, DataLoader

    return config, CSPDiffusion, Data, DataLoader


def init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed refinement requires CUDA.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def lattices_to_params_shape(lattices: torch.Tensor):
    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        cos_angle = torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1) / (
            lengths[..., j] * lengths[..., k]
        )
        angles[..., i] = torch.clamp(cos_angle, -1.0, 1.0)
    return lengths, torch.arccos(angles) * 180.0 / np.pi


class ProposalDataset(Dataset):
    def __init__(self, proposal_graphs: List[Dict], Data):
        self.proposal_graphs = proposal_graphs
        self.Data = Data

    def __len__(self) -> int:
        return len(self.proposal_graphs)

    def __getitem__(self, index):
        s = self.proposal_graphs[index]
        n_atom = int(torch.as_tensor(s["n_atom"]).view(-1)[0].item())
        return self.Data(
            num_atoms=torch.LongTensor([n_atom]),
            num_nodes=n_atom,
            num_bonds=s["edge_indices"].shape[0],
            lengths=torch.as_tensor(s["length"], dtype=torch.float32).view(1, 3),
            angles=torch.as_tensor(s["angle"], dtype=torch.float32).view(1, 3),
            frac_coords=torch.as_tensor(s["x_coord"], dtype=torch.float32),
            atom_types=torch.LongTensor(s["a_type"]),
            edge_index=torch.LongTensor(s["edge_indices"].T).contiguous(),
            to_jimages=torch.LongTensor(s["to_jimages"]),
        )


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def empty_payload(num_evals: int) -> Dict[str, torch.Tensor | float]:
    return {
        "frac_coords": torch.empty((num_evals, 0, 3), dtype=torch.float32),
        "num_atoms": torch.empty((num_evals, 0), dtype=torch.long),
        "atom_types": torch.empty((num_evals, 0), dtype=torch.long),
        "lengths": torch.empty((num_evals, 0, 3), dtype=torch.float32),
        "angles": torch.empty((num_evals, 0, 3), dtype=torch.float32),
        "time": 0.0,
    }


def merge_distributed_outputs(output_dir: Path, world_size: int, total_proposals: int, diff_steps: int, num_evals: int) -> None:
    payloads: List[Dict[str, Any]] = []
    assigned_total = 0
    wall_time = 0.0
    for rank in range(world_size):
        metrics_path = output_dir / f"refinement_metrics.rank{rank}.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assigned_total += int(metrics.get("assigned_proposals", 0))
        wall_time = max(wall_time, float(metrics.get("time_sec") or 0.0))
        output_file = Path(metrics["output_file"])
        if output_file.exists():
            payloads.append(torch.load(output_file, map_location="cpu"))

    if payloads:
        payload = {
            "frac_coords": torch.cat([item["frac_coords"] for item in payloads], dim=1),
            "num_atoms": torch.cat([item["num_atoms"] for item in payloads], dim=1),
            "atom_types": torch.cat([item["atom_types"] for item in payloads], dim=1),
            "lengths": torch.cat([item["lengths"] for item in payloads], dim=1),
            "angles": torch.cat([item["angles"] for item in payloads], dim=1),
            "time": wall_time,
        }
    else:
        payload = empty_payload(num_evals)
        payload["time"] = wall_time

    output_file = output_dir / f"dlm_refined_mp_{assigned_total}.pt"
    torch.save(payload, output_file)
    write_json(
        output_dir / "refinement_metrics.json",
        {
            "num_proposals": total_proposals,
            "assigned_proposals": assigned_total,
            "output_file": str(output_file),
            "time_sec": wall_time,
            "diff_steps": diff_steps,
            "num_evals": num_evals,
            "distributed": True,
            "world_size": world_size,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-graphs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--diff-steps", type=int, default=800)
    parser.add_argument("--num-evals", type=int, default=1)
    parser.add_argument("--run-type", default="train")
    parser.add_argument("--max-proposals", type=int, default=1000)
    parser.add_argument("--metadata-jsonl", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    if is_main:
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run_config["distributed"] = distributed
        run_config["world_size"] = world_size
        write_json(args.output_dir / "run_config.json", run_config)
        with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "not_applicable",
                        "stage": "diffusion_refinement",
                        "reason": "refinement run; no optimizer training is performed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    config, CSPDiffusion, Data, DataLoader = setup_crysllmgen_imports(args.crysllmgen_dir)
    device = dist_info["device"]

    proposal_graphs = torch.load(args.proposal_graphs, map_location="cpu")
    proposal_metadata = None
    if args.metadata_jsonl is not None:
        with args.metadata_jsonl.open(encoding="utf-8") as handle:
            proposal_metadata = [json.loads(line) for line in handle if line.strip()]
        if len(proposal_metadata) != len(proposal_graphs):
            raise ValueError(
                f"metadata rows {len(proposal_metadata)} != proposal graphs {len(proposal_graphs)}"
            )
    if args.max_proposals:
        proposal_graphs = proposal_graphs[: args.max_proposals]
        if proposal_metadata is not None:
            proposal_metadata = proposal_metadata[: args.max_proposals]
    total_proposals = len(proposal_graphs)
    rank_graphs = proposal_graphs[rank::world_size] if distributed else proposal_graphs
    rank_metadata = None if proposal_metadata is None else (
        proposal_metadata[rank::world_size] if distributed else proposal_metadata
    )
    if rank_metadata is not None and int(args.batch_size) != 1:
        raise ValueError("--metadata-jsonl requires --batch-size 1 for per-proposal refiner seeds")
    dataset = ProposalDataset(rank_graphs, Data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = CSPDiffusion(args.timesteps, args.run_type).to(device)
    model.device = device
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    frac_coords_all, num_atoms_all, atom_types_all, lattices_all = [], [], [], []
    start = time.time()
    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc=f"CrysLLMGen refinement rank{rank}", disable=distributed and not is_main)
        ):
            batch = batch.to(device)
            if rank_metadata is not None and "refiner_seed" in rank_metadata[batch_index]:
                refiner_seed = int(rank_metadata[batch_index]["refiner_seed"])
                torch.manual_seed(refiner_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(refiner_seed)
            batch_frac, batch_num, batch_atom, batch_lat = [], [], [], []
            for _ in range(args.num_evals):
                outputs, _ = model.sample(batch, diff_steps=args.diff_steps)
                batch_frac.append(outputs["frac_coords"].detach().cpu())
                batch_num.append(outputs["num_atoms"].detach().cpu())
                batch_atom.append(outputs["atom_types"].detach().cpu())
                batch_lat.append(outputs["lattices"].detach().cpu())
            frac_coords_all.append(torch.stack(batch_frac))
            num_atoms_all.append(torch.stack(batch_num))
            atom_types_all.append(torch.stack(batch_atom))
            lattices_all.append(torch.stack(batch_lat))

    elapsed = time.time() - start
    output_file = (
        args.output_dir / f"dlm_refined_mp_{len(rank_graphs)}.rank{rank}.pt"
        if distributed
        else args.output_dir / f"dlm_refined_mp_{len(rank_graphs)}.pt"
    )
    if frac_coords_all:
        frac_coords = torch.cat(frac_coords_all, dim=1)
        num_atoms = torch.cat(num_atoms_all, dim=1)
        atom_types = torch.cat(atom_types_all, dim=1)
        lattices = torch.cat(lattices_all, dim=1)
        lengths, angles = lattices_to_params_shape(lattices)
        payload = {
            "frac_coords": frac_coords,
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lengths": lengths,
            "angles": angles,
            "time": elapsed,
        }
    else:
        payload = empty_payload(args.num_evals)
        payload["time"] = elapsed
    torch.save(payload, output_file)

    if rank_metadata is not None:
        metadata_name = f"refined_metadata.rank{rank}.jsonl" if distributed else "refined_metadata.jsonl"
        with (args.output_dir / metadata_name).open("w", encoding="utf-8") as handle:
            for proposal_index, row in enumerate(rank_metadata):
                enriched = dict(row)
                enriched.update({"proposal_index": proposal_index, "num_evals": int(args.num_evals), "refined": True})
                handle.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")

    metrics_name = f"refinement_metrics.rank{rank}.json" if distributed else "refinement_metrics.json"
    write_json(
        args.output_dir / metrics_name,
        {
            "num_proposals": total_proposals,
            "assigned_proposals": len(rank_graphs),
            "output_file": str(output_file),
            "time_sec": elapsed,
            "diff_steps": args.diff_steps,
            "num_evals": args.num_evals,
            "distributed": distributed,
            "rank": rank,
            "world_size": world_size,
        },
    )
    if is_main:
        with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "refinement_complete",
                        "assigned_proposals": len(rank_graphs),
                        "time_sec": elapsed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size, total_proposals, args.diff_steps, args.num_evals)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

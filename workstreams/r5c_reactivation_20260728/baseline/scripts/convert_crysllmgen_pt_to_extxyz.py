#!/usr/bin/env python3
"""Convert CrysLLMGen-style crystal tensors to an extxyz trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase.io import write
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def as_crystal_records(data: dict) -> list[dict]:
    required = ["frac_coords", "atom_types", "lengths", "angles", "num_atoms"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required keys in input pt file: {missing}")

    frac_coords = data["frac_coords"]
    atom_types = data["atom_types"]
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms = data["num_atoms"]

    if frac_coords.ndim == 3:
        if frac_coords.shape[0] != 1:
            raise ValueError(f"Expected unbatched or batch-size-1 tensors, got {frac_coords.shape}")
        frac_coords = frac_coords[0]
        atom_types = atom_types[0]
        lengths = lengths[0]
        angles = angles[0]
        num_atoms = num_atoms[0]

    records: list[dict] = []
    start = 0
    for index, count_value in enumerate(num_atoms.detach().cpu().tolist()):
        count = int(count_value)
        stop = start + count
        records.append(
            {
                "frac_coords": frac_coords[start:stop].detach().cpu().numpy(),
                "atom_types": atom_types[start:stop].detach().cpu().numpy().astype(int).tolist(),
                "lengths": lengths[index].detach().cpu().numpy().tolist(),
                "angles": angles[index].detach().cpu().numpy().tolist(),
            }
        )
        start = stop
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pt", type=Path, required=True)
    parser.add_argument("--output-extxyz", type=Path, required=True)
    parser.add_argument("--max-structures", type=int, default=None)
    args = parser.parse_args()

    data = torch.load(args.input_pt, map_location="cpu")
    records = as_crystal_records(data)
    if args.max_structures is not None:
        records = records[: args.max_structures]

    adaptor = AseAtomsAdaptor()
    atoms_list = []
    for idx, record in enumerate(records):
        structure = Structure(
            lattice=Lattice.from_parameters(*record["lengths"], *record["angles"]),
            species=record["atom_types"],
            coords=record["frac_coords"],
            coords_are_cartesian=False,
        )
        atoms = adaptor.get_atoms(structure)
        atoms.info["crystal_index"] = idx
        atoms_list.append(atoms)

    args.output_extxyz.parent.mkdir(parents=True, exist_ok=True)
    write(args.output_extxyz, atoms_list, format="extxyz")
    print(f"WROTE {args.output_extxyz} n_structures={len(atoms_list)}")


if __name__ == "__main__":
    main()

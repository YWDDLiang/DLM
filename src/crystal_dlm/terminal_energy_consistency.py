"""A uniform terminal-energy reproducibility check, not a physical oracle."""
from __future__ import annotations

import math
import numpy as np

TERMINAL_VERIFICATION_PROTOCOL = {
    "fresh_terminal_energy": True, "periodic_coordinate_wrap": True,
    "rigid_fractional_shift": [.137, .271, .419], "energy_tolerance_eV_atom": .001,
}


def compare_energies(stored, fresh, *, tolerance=.001):
    values = [float(stored), *map(float, fresh)]
    if len(fresh) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("three finite terminal representation energies are required")
    delta = fresh[0] - stored
    spread = max(fresh) - min(fresh)
    return {"stored_energy_matches_fresh": abs(delta) <= tolerance,
            "periodic_representation_consistent": spread <= tolerance,
            "fresh_minus_stored_eV_atom": delta, "representation_spread_eV_atom": spread,
            "status": "consistent" if abs(delta) <= tolerance and spread <= tolerance else "inconsistent"}


def check_terminal_energy(model, original, stored_energy):
    from pymatgen.core import Structure
    wrapped = Structure(original.lattice, original.species, np.mod(original.frac_coords, 1.))
    shifted = Structure(original.lattice, original.species, np.mod(original.frac_coords + [.137, .271, .419], 1.))
    scores = []
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)
    for name, structure in (("stored_geometry", original), ("wrapped_geometry", wrapped), ("shifted_geometry", shifted)):
        result = model.predict_structure(structure, task="efs")
        if isinstance(result, list):
            result = result[0]
        energy = float(result["e"])
        forces, stress = array(result["f"]), array(result["s"])
        if not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError("nonfinite fresh force or stress")
        scores.append({"representation": name, "energy_eV_atom": energy,
                       "force_max_eV_A": float(np.linalg.norm(forces, axis=-1).max()),
                       "stress_max_GPa": float(np.abs(stress).max())})
    return {"stored_terminal_energy_eV_atom": stored_energy, "scores": scores,
            **compare_energies(stored_energy, [r["energy_eV_atom"] for r in scores])}

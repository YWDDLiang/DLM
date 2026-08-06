from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PYMATGEN_AVAILABLE = importlib.util.find_spec("pymatgen") is not None


@unittest.skipUnless(PYMATGEN_AVAILABLE, "pymatgen phase diagram is tested locally/server")
class HullClosureTests(unittest.TestCase):
    def test_toy_reference_requires_relaxation_then_closes(self) -> None:
        from crystal_dlm.wqcodiff.hull import (
            build_hull_closure_step,
            load_frozen_hull,
        )

        with tempfile.TemporaryDirectory() as directory:
            energies = Path(directory) / "energies.jsonl"
            raw = [
                ("li", {"Li": 1}, -1.0),
                ("o", {"O": 1}, -2.0),
                ("li2o", {"Li": 2, "O": 1}, -5.0),
            ]
            rows = []
            for reference_id, composition, energy in raw:
                rows.append(
                    {
                        "schema": "wqcodiff_reference_energy_v1",
                        "reference_id": reference_id,
                        "evaluator": "toy",
                        "contract_hash": "contract",
                        "stage": "raw",
                        "status": "succeeded",
                        "structure_hash": reference_id,
                        "composition": composition,
                        "energy_total_ev": energy,
                        "energy_per_atom_ev": energy / sum(composition.values()),
                    }
                )
            energies.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            round0 = build_hull_closure_step(
                [energies],
                output_path=Path(directory) / "round0.json",
                round_index=0,
                expected_reference_count=3,
            )
            self.assertEqual(round0["pending_count"], 3)
            with energies.open("a", encoding="utf-8") as handle:
                for row in rows:
                    relaxed = dict(row)
                    relaxed["stage"] = "relaxed"
                    relaxed["energy_total_ev"] -= 0.1
                    relaxed["energy_per_atom_ev"] = relaxed["energy_total_ev"] / sum(
                        relaxed["composition"].values()
                    )
                    handle.write(json.dumps(relaxed) + "\n")
            final_path = Path(directory) / "final.json"
            final = build_hull_closure_step(
                [energies],
                output_path=final_path,
                round_index=1,
                expected_reference_count=3,
            )
            self.assertTrue(final["gate_passed"])
            payload, phase_diagram = load_frozen_hull(final_path)
            self.assertEqual(payload["hull_sha256"], final["hull_sha256"])
            self.assertEqual(len(phase_diagram.all_entries), 3)


if __name__ == "__main__":
    unittest.main()

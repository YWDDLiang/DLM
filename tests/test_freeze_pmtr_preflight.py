import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


SPEC = importlib.util.spec_from_file_location(
    "freeze_pmtr_preflight", ROOT / "scripts" / "freeze_pmtr_preflight.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import freeze_pmtr_preflight")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def body(offset=0.0):
    answer, _ = arrays_to_dynamic_answer(
        lengths=[4.0, 4.0, 4.0],
        angles=[90.0, 90.0, 90.0],
        species=["Li", "O"],
        frac_coords=[[0.0 + offset, 0.0, 0.0], [0.5, 0.5, 0.5]],
        separator="",
    )
    return answer


class FreezePMTRPreflightTest(unittest.TestCase):
    def test_freeze_is_deterministic_disjoint_and_outcome_blind(self):
        teacher = [
            {
                "source_row_idx": index,
                "prompt": "p",
                "answer": body(),
                "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
            }
            for index in range(12)
        ]
        pointer = [
            {"source_row_idx": index, "species_program": ["Li", "O"]}
            for index in range(12)
        ]
        states = [
            {
                "mp20_train_source_row_idx": index,
                "sample_idx": index,
                "prompt": "p",
                "final_body": body(0.01),
                "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
                "species_program": ["Li", "O"],
                "outcomes_read": False,
                "replacement": False,
            }
            for index in range(6)
        ]
        first = MODULE.freeze(
            teacher_rows=teacher,
            pointer_rows=pointer,
            actual_states=states,
            seed=7,
            fit_size=4,
            holdout_size=2,
            transfer_size=2,
        )
        second = MODULE.freeze(
            teacher_rows=teacher,
            pointer_rows=pointer,
            actual_states=states,
            seed=7,
            fit_size=4,
            holdout_size=2,
            transfer_size=2,
        )
        self.assertEqual(first["fit_ids"], second["fit_ids"])
        groups = [set(first[name]) for name in ("fit_ids", "holdout_ids", "transfer_ids")]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertTrue(all(row["outcomes_read"] is False for row in first["transfer"]))

    def test_cli_writes_compact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher.jsonl"
            pointer = root / "pointer.jsonl"
            states = root / "states.jsonl"
            teacher.write_text(
                "".join(
                    json.dumps(
                        {
                            "source_row_idx": index,
                            "prompt": "p",
                            "answer": body(),
                            "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
                        }
                    )
                    + "\n"
                    for index in range(5)
                )
            )
            pointer.write_text(
                "".join(
                    json.dumps({"source_row_idx": index, "species_program": ["Li", "O"]}) + "\n"
                    for index in range(5)
                )
            )
            states.write_text(
                json.dumps(
                    {
                        "mp20_train_source_row_idx": 0,
                        "prompt": "p",
                        "final_body": body(0.01),
                        "plan_state": {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]},
                        "species_program": ["Li", "O"],
                        "outcomes_read": False,
                        "replacement": False,
                    }
                )
                + "\n"
            )
            output = root / "frozen"
            MODULE.main(
                [
                    "--teacher-train", str(teacher),
                    "--pointer-train", str(pointer),
                    "--actual-spad-states", str(states),
                    "--output-dir", str(output),
                    "--fit-size", "2",
                    "--holdout-size", "1",
                    "--transfer-size", "1",
                ]
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["pairwise_disjoint"])
            self.assertFalse(manifest["outcomes_read"])
            self.assertTrue((output / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()

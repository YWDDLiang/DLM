import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_raw1000_two500",
    PROJECT_ROOT / "scripts" / "freeze_raw1000_two500.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
freeze = MODULE.freeze


def plan(index: int) -> dict:
    return {
        "formula": f"Li{index + 1}O",
        "N": 2,
        "elements": ["Li", "O"],
        "counts": [1, 1],
        "anion_framework": "oxide",
        "charge_bucket": "neutral_plausible",
        "lattice_system": "cubic",
        "spacegroup_bucket": "sg_195_230",
        "volume_per_atom_bin": "volpa_010_014",
    }


class FreezeRaw1000Test(unittest.TestCase):
    def test_first_and_last_valid_500_preserve_global_ordinals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"parsed": False, "sample_idx": 99}) + "\n")
                for index in range(1003):
                    handle.write(
                        json.dumps(
                            {"sample_idx": 1000 + index, "plan_state": plan(index)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            output = root / "frozen"
            manifest = freeze(source, output)
            self.assertEqual(manifest["frozen_rows"], 1000)
            self.assertEqual(manifest["invalid_rows_skipped"], 1)
            first = [json.loads(line) for line in (output / "round1_first500.jsonl").read_text().splitlines()]
            last = [json.loads(line) for line in (output / "round2_last500.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_idx"] for row in first], list(range(500)))
            self.assertEqual([row["sample_idx"] for row in last], list(range(500, 1000)))
            self.assertEqual(first[0]["source_sample_idx"], 1000)
            self.assertEqual(last[-1]["source_sample_idx"], 1999)
            self.assertTrue((output / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()
